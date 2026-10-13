"""Rebuild conference tables and vector figures from frozen evidence only."""
from pathlib import Path
from datetime import date
import sys
import json
import csv
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/figure_qa'))
from audit_panel_alignment import require_matplotlib_panel_alignment

COLORS = {'cma': '#006D77', 'base': '#697687', 'lear': '#B86B31', 'pale': '#DCEEF0'}
SEEDS = [11, 22, 42, 77, 123]
MARKETS = ['DE_LU', 'FR', 'NL', 'BE']
DISPLAY = ['DE-LU', 'FR', 'NL', 'BE']
LABELS = {
    'cma_former': 'CMA-Former', 'lear': 'LEAR', 'patchtst': 'PatchTST (adapted)',
    'temporal_gat': 'Temporal-GAT', 'concatenated_transformer': 'Concat. Transformer',
    'single_market_patchtst': 'Separate PatchTST', 'itransformer': 'iTransformer (adapted)',
    'naive_previous_day': 'Previous-day curve', 'cma_no_cross': 'No market exchange',
    'cma_late_future': 'Output-only context', 'cma_topology_prior': 'Soft topology',
    'cma_fixed_prior': 'Fixed topology', 'cma_shuffled_prior': 'Shuffled topology',
}
MAIN = ['cma_former', 'lear', 'patchtst', 'single_market_patchtst',
        'concatenated_transformer', 'itransformer', 'temporal_gat', 'naive_previous_day']
ABLATIONS = ['cma_former', 'cma_no_cross', 'cma_late_future',
             'cma_topology_prior', 'cma_fixed_prior', 'cma_shuffled_prior']


def collect():
    rows, markets = [], []
    gpu = ROOT / 'source_data/applied_energy_final'
    for p in sorted(gpu.glob('*/seed_*/metrics.json')):
        model, seed = p.parent.parent.name, int(p.parent.name[5:])
        data = json.loads(p.read_text())
        diag = json.loads((p.parent / 'model_diagnostics.json').read_text())
        rows.append(dict(model=model, seed=seed, parameters=diag['trainable_parameters'], **data['macro']))
        markets.extend(dict(model=model, seed=seed, market=m, **data['per_market'][m]) for m in MARKETS)
    for model in ['lear', 'naive_previous_day']:
        data = json.loads((ROOT / f'source_data/applied_energy_final_cpu/{model}/metrics.json').read_text())
        rows.append(dict(model=model, seed=np.nan, parameters=np.nan, **data['macro']))
        markets.extend(dict(model=model, seed=np.nan, market=m, **data['per_market'][m]) for m in MARKETS)
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'source_data/metrics_all_seeds.csv', index=False)
    pd.DataFrame(markets).to_csv(ROOT / 'source_data/metrics_by_market.csv', index=False)
    for model in set(MAIN + ABLATIONS) - {'lear', 'naive_previous_day'}:
        assert sorted(frame.loc[frame.model.eq(model), 'seed'].astype(int)) == SEEDS
    return frame


def style():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
        'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.linewidth': .65, 'lines.linewidth': 1.1,
        'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.fonttype': 'none', 'figure.facecolor': 'white',
        'savefig.facecolor': 'white', 'axes.axisbelow': True,
    })


def save(fig, name):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,
        json_out=str(ROOT / f'audit/{name}.alignment.json'),
        overlay_svg=str(ROOT / f'audit/{name}.alignment.svg'), strict=True)
    fig.savefig(ROOT / f'figures/{name}.pdf')
    fig.savefig(ROOT / f'figures/{name}.svg')
    fig.savefig(ROOT / f'audit/{name}_preview.png', dpi=300)
    plt.close(fig)


def framework():
    from draw_framework import render_framework
    render_framework(save)


def tables(frame):
    grouped = frame.groupby('model')
    summary = grouped[['mae','rmse','neg_f1','neg_precision','neg_recall','pinball','coverage_80','parameters']].agg(['mean','std'])
    summary.to_csv(ROOT / 'source_data/summary_mean_sd.csv')
    def value(model, metric, digits):
        mean, sd = summary.loc[model, (metric,'mean')], summary.loc[model,(metric,'std')]
        a = f'{mean:.{digits}f}'
        if model == 'cma_former' and metric in ['mae','rmse']:
            a = r'\mathbf{' + a + '}'
        if model == 'lear' and metric == 'neg_f1':
            a = r'\mathbf{' + a + '}'
        return '$' + a + (rf'\!\pm\!{sd:.{digits}f}' if np.isfinite(sd) else '') + '$'
    lines = [r'\begin{table}[!t]',r'\centering',
        r'\caption{Test errors (EUR/MWh) and hourly negative-price F1. Mean $\pm$ sample SD over five seeds; deterministic controls have no SD. All scores are macro-averaged across four markets.}',
        r'\label{tab:main}',r'\footnotesize',r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{@{}lccc@{}}',r'\toprule',
        r'Model & MAE $\downarrow$ & RMSE $\downarrow$ & F1 $\uparrow$ \\',r'\midrule']
    for model in MAIN:
        label = LABELS[model].replace(' (adapted)',r'$^{\dagger}$')
        lines.append(label+' & '+' & '.join(value(model,k,d) for k,d in [('mae',2),('rmse',2),('neg_f1',3)])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',
              r'\par\smallskip\begin{minipage}{\columnwidth}\scriptsize $^{\dagger}$Task-adapted implementations with the common future-covariate adapter; not reported scores from the original papers.\end{minipage}',r'\end{table}']
    (ROOT/'tables/table1_results.tex').write_text('\n'.join(lines)+'\n',encoding='ascii')
    macros = {'CmaMAE':('cma_former','mae',2),'CmaSD':('cma_former','mae',2),
              'CmaRMSE':('cma_former','rmse',2),'LearMAE':('lear','mae',2),
              'PatchMAE':('patchtst','mae',2),'NoCrossMAE':('cma_no_cross','mae',2),
              'LateMAE':('cma_late_future','mae',2)}
    output=[]
    for name,(model,metric,digits) in macros.items():
        val=summary.loc[model,(metric,'std' if name=='CmaSD' else 'mean')]
        output.append('\\newcommand{\\'+name+'}{'+f'{val:.{digits}f}'+'}')
    for name,base in [('PatchGain','patchtst'),('LearGain','lear'),('CrossGain','cma_no_cross')]:
        val=100*(1-summary.loc['cma_former',('mae','mean')]/summary.loc[base,('mae','mean')])
        output.append('\\newcommand{\\'+name+'}{'+f'{val:.1f}'+'}')
    (ROOT/'results_macros.tex').write_text('\n'.join(output)+'\n',encoding='ascii')
    return summary


def ablations(frame,summary):
    fig,ax=plt.subplots(figsize=(3.49,2.12))
    fig.subplots_adjust(left=.405,right=.97,bottom=.22,top=.89)
    for y,model in enumerate(ABLATIONS):
        color=COLORS['cma'] if model=='cma_former' else COLORS['base']
        values=frame.loc[frame.model.eq(model),'mae'].to_numpy()
        mean=values.mean();sd=values.std(ddof=1)
        ax.scatter(values,np.full(5,y)+np.linspace(-.14,.14,5),s=9,color=color,alpha=.45,zorder=3)
        ax.errorbar(mean,y,xerr=sd,fmt='D',ms=4,color=color,capsize=2,lw=1.1,zorder=4)
    ax.set_yticks(range(6),[LABELS[x] for x in ABLATIONS])
    ax.set_ylim(5.65,-.65)
    ax.set_xlim(14,19)
    ax.set_xticks([14,16,18])
    ax.set_xlabel('MAE (EUR/MWh)',labelpad=3)
    ax.grid(axis='x',color='#E4E8EC',lw=.5)
    ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)
    ax.text(0,1.07,'Architecture and topology controls',transform=ax.transAxes,ha='left',fontsize=8,weight='bold')
    save(fig,'fig2_ablation')


def market_differences():
    data=pd.read_csv(ROOT/'source_data/aggregate_daily_dm_and_block_bootstrap.csv')
    selected=data[data.other.isin(['patchtst','lear'])].copy()
    selected.to_csv(ROOT/'source_data/fig3_market_intervals.csv',index=False)
    fig,ax=plt.subplots(figsize=(3.49,2.68))
    fig.subplots_adjust(left=.16,right=.965,bottom=.19,top=.86)
    for model,shift,color,label in [('patchtst',-.13,COLORS['base'],'vs. PatchTST'),('lear',.13,COLORS['lear'],'vs. LEAR')]:
        d=selected[selected.other.eq(model)].sort_values('market_index')
        x=d.mean_loss_diff.to_numpy()
        low=d['bootstrap_ci_2.5'].to_numpy();high=d['bootstrap_ci_97.5'].to_numpy()
        ax.errorbar(x,d.market_index+shift,xerr=[x-low,high-x],fmt='o',ms=3.5,
                    color=color,capsize=2,label=label,lw=1.05)
    ax.axvline(0,color='#555555',lw=.7,ls='--',zorder=0)
    ax.set_yticks(range(4),DISPLAY)
    ax.set_ylim(3.6,-.6)
    ax.set_xlim(-5,1.7)
    ax.set_xticks([-4,-2,0,1])
    ax.set_xlabel('CMA-Former minus comparator MAE (EUR/MWh)',fontsize=7,labelpad=4)
    ax.legend(loc='lower left',bbox_to_anchor=(-.03,1.04),ncol=2,handlelength=1.3,columnspacing=1.0,borderaxespad=0)
    ax.tick_params(axis='y',length=0)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x',color='#E4E8EC',lw=.5)
    save(fig,'fig3_market_differences')


def verify_predictions(frame):
    checks=[]
    truth=None;dates=None
    for model in ['cma_former','patchtst','lear']:
        name = model + ('_seed42' if model!='lear' else '')
        a=np.load(ROOT/f'source_data/predictions/{name}.npz')
        y=a['y_true_raw'];p=a['y_pred_raw']
        assert y.shape==p.shape==(544,4,24)
        if truth is None:
            truth=y;dates=a['delivery_date_ordinal']
        else:
            np.testing.assert_array_equal(y,truth)
            np.testing.assert_array_equal(a['delivery_date_ordinal'],dates)
        errors=np.abs(y-p)
        mae=float(np.mean([errors[:,m].mean() for m in range(4)]))
        row=frame[frame.model.eq(model)]
        if model!='lear':row=row[row.seed.eq(42)]
        reference=float(row.iloc[0].mae)
        assert abs(mae-reference)<1e-4,(model,mae,reference)
        checks.append(dict(model=model,shape=list(y.shape),mae_recomputed=mae,mae_stored=reference))
    (ROOT/'audit/prediction_verification.json').write_text(json.dumps(checks,indent=2))


def forecast_cases():
    models=['cma_former','patchtst','lear']
    arrays={m:np.load(ROOT/f'source_data/predictions/{m + ("_seed42" if m!="lear" else "")}.npz') for m in models}
    truth=arrays['cma_former']['y_true_raw']
    negative=(truth<0).any(axis=2)
    daily_range=np.ptp(truth.astype(float),axis=2)
    fig,axes=plt.subplots(2,1,figsize=(3.49,2.68),sharex=True)
    fig.subplots_adjust(left=.18,right=.97,bottom=.16,top=.83,hspace=.58)
    rows=[];selections=[]
    for ax,is_negative,panel in zip(axes,[False,True],['a','b']):
        candidates=np.argwhere(negative==is_negative)
        median=float(np.median(daily_range[negative==is_negative]))
        # argwhere is in day-then-market order, which also breaks ties.
        k=np.argmin([abs(daily_range[d,m]-median) for d,m in candidates])
        day,market=map(int,candidates[k])
        dstr=date.fromordinal(int(arrays['cma_former']['delivery_date_ordinal'][day])).isoformat()
        selections.append(dict(category='negative' if is_negative else 'nonnegative',day_index=day,
            market=MARKETS[market],delivery_date=dstr,category_median_range=median,
            selected_range=float(daily_range[day,market]),neural_seed=42))
        ax.plot(range(24),truth[day,market],color='#202020',lw=1.35,label='Observed',zorder=4)
        for model,color,ls,label in [('cma_former',COLORS['cma'],'-','CMA-Former'),
                ('patchtst',COLORS['base'],'--','PatchTST'),('lear',COLORS['lear'],':','LEAR')]:
            ax.plot(range(24),arrays[model]['y_pred_raw'][day,market],color=color,ls=ls,lw=1.1,label=label)
        ax.axhline(0,color='#B7BEC5',lw=.6,zorder=0)
        ax.grid(axis='y',color='#E7EBEE',lw=.4)
        ax.set_xlim(0,23)
        ax.set_xticks([0,6,12,18,23])
        ax.set_ylabel('EUR/MWh',fontsize=7)
        ax.text(0,1.08,f'({panel}) {DISPLAY[market]} | {dstr}',transform=ax.transAxes,fontsize=7.5,weight='bold')
        for hour in range(24):
            rows.append(dict(panel=panel,delivery_date=dstr,market=MARKETS[market],hour=hour,
                observed=float(truth[day,market,hour]),**{m:float(arrays[m]['y_pred_raw'][day,market,hour]) for m in models}))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.56,.998),ncol=2,
               columnspacing=1.5,handlelength=1.9,borderaxespad=0)
    axes[1].set_xlabel('Delivery hour (local time)',fontsize=7.5,labelpad=2)
    pd.DataFrame(rows).to_csv(ROOT/'source_data/fig4_forecast_curves.csv',index=False)
    (ROOT/'audit/forecast_case_selection.json').write_text(json.dumps(dict(
        rule='Within nonnegative and negative market-days separately, select the day-market pair whose observed intraday range is nearest the category median. Ties: chronological then fixed market order. Selection never uses forecasts.',
        selections=selections),indent=2))
    save(fig,'fig4_forecast_cases')


if __name__=='__main__':
    for folder in ['figures','tables','audit','source_data']:(ROOT/folder).mkdir(exist_ok=True)
    style()
    frame=collect()
    summary=tables(frame)
    verify_predictions(frame)
    framework()
    ablations(frame,summary)
    market_differences()
    forecast_cases()
    print('Rebuilt 4 vector figures, table, macros, and evidence CSVs. No training or new inference.')
