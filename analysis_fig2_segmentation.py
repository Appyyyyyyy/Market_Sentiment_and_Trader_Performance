import pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
warnings.filterwarnings('ignore')

BG='#0d1117'; PANEL='#161b22'; BORDER='#30363d'; TEXT='#c9d1d9'; MUTED='#8b949e'
FEAR_C='#f85149'; GREED_C='#3fb950'; NEUTRAL_C='#d29922'
EGREED_C='#56d364'; EFEAR_C='#ff6b6b'; ACCENT1='#79c0ff'; ACCENT2='#d2a8ff'; ACCENT3='#ffa657'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':PANEL,'axes.edgecolor':BORDER,
    'axes.labelcolor':TEXT,'xtick.color':MUTED,'ytick.color':MUTED,'text.color':TEXT,
    'grid.color':BORDER,'grid.alpha':0.7,'font.family':'DejaVu Sans','font.size':9.5,
    'axes.titlesize':11,'axes.titleweight':'bold','legend.facecolor':PANEL,'legend.edgecolor':BORDER,
    'axes.spines.top':False,'axes.spines.right':False})

trades = pd.read_csv('/mnt/user-data/uploads/historical_data__1_.csv')
fg     = pd.read_csv('/mnt/user-data/uploads/fear_greed_index__1_.csv')
trades['date'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True).dt.normalize()
fg['date']     = pd.to_datetime(fg['date'])
fg['simple_sent'] = fg['classification'].map({'Extreme Fear':'Fear/Extreme Fear','Fear':'Fear/Extreme Fear','Neutral':'Neutral','Greed':'Greed/Extreme Greed','Extreme Greed':'Greed/Extreme Greed'})
merged = trades.merge(fg[['date','classification','value','simple_sent']], on='date', how='left')
merged.rename(columns={'classification':'sentiment','value':'fg_value'}, inplace=True)
closes = merged[merged['Closed PnL'] != 0].copy()
closes['is_win'] = closes['Closed PnL'] > 0
closes['net_pnl'] = closes['Closed PnL'] - closes['Fee']
closes['month']  = closes['date'].dt.to_period('M')
trader = closes.groupby('Account').agg(
    total_pnl=('Closed PnL','sum'), net_pnl=('net_pnl','sum'),
    n_trades=('Closed PnL','count'), win_rate=('is_win','mean'),
    avg_size=('Size USD','mean'), pnl_std=('Closed PnL','std'), total_fees=('Fee','sum'),
).reset_index()
trader['avg_pnl_per_trade'] = trader['total_pnl'] / trader['n_trades']
trader['sharpe'] = trader['total_pnl'] / (trader['pnl_std'] + 1e-6)
trader['freq_bucket'] = pd.qcut(trader['n_trades'],3,labels=['Infrequent','Medium','Frequent'])
trader['size_bucket']  = pd.qcut(trader['avg_size'],3,labels=['Small','Medium','Large'])
trader['winner'] = trader['total_pnl'] > 0
order = ['Fear/Extreme Fear','Neutral','Greed/Extreme Greed']

fig2 = plt.figure(figsize=(20, 22), facecolor=BG)
gs2 = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.48, wspace=0.35,
                        top=0.95, bottom=0.04, left=0.07, right=0.97)
fig2.text(0.5, 0.975, 'Trader Segmentation  —  Who Wins, Who Loses, and Why',
          ha='center', fontsize=17, fontweight='bold', color='#f0f6fc')

# scatter PnL vs win rate
ax = fig2.add_subplot(gs2[0, :2])
scatter_c = [GREED_C if w else FEAR_C for w in trader['winner']]
sizes = np.clip(trader['n_trades']/50, 30, 500)
ax.scatter(trader['win_rate']*100, trader['total_pnl']/1000, s=sizes, c=scatter_c, alpha=0.75, edgecolors='none')
for _, row in trader.nlargest(5,'total_pnl').iterrows():
    ax.annotate(f"…{row['Account'][-6:]}", (row['win_rate']*100, row['total_pnl']/1000),
                fontsize=8, color=TEXT, xytext=(5,3), textcoords='offset points')
ax.axhline(0, color=BORDER, lw=1); ax.axvline(50, color=BORDER, lw=1)
ax.set_xlabel('Win Rate (%)'); ax.set_ylabel('Total PnL (USD K)')
ax.set_title('Win Rate vs Total PnL  (bubble size = trade count  |  🟢 Winner  🔴 Loser)')

# trader ranking
ax2b = fig2.add_subplot(gs2[0, 2])
tr_s = trader.sort_values('total_pnl')
ax2b.barh(range(len(tr_s)), tr_s['total_pnl']/1000,
          color=[GREED_C if p>0 else FEAR_C for p in tr_s['total_pnl']], edgecolor=BG, height=0.75)
ax2b.axvline(0, color='white', lw=0.8)
ax2b.set_xlabel('Total PnL (USD K)'); ax2b.set_title('All 32 Traders Ranked\nby Total PnL'); ax2b.set_yticks([])

# freq bucket
ax3 = fig2.add_subplot(gs2[1, 0])
freq_stats = trader.groupby('freq_bucket', observed=True).agg(mean_pnl=('total_pnl','mean'),count=('Account','count')).reset_index()
bars = ax3.bar([f"{r['freq_bucket']}\n(n={r['count']})" for _,r in freq_stats.iterrows()],
               freq_stats['mean_pnl']/1000, color=[ACCENT1,ACCENT2,ACCENT3], edgecolor=BG)
for bar, val in zip(bars, freq_stats['mean_pnl'].values):
    ax3.text(bar.get_x()+bar.get_width()/2, (bar.get_height() if val>0 else bar.get_height())-8,
             f'${val/1000:.0f}K', ha='center', fontsize=9.5, color='white', fontweight='bold')
ax3.axhline(0, color=BORDER, lw=1)
ax3.set_title('Avg PnL by Frequency\nSegment'); ax3.set_ylabel('Mean Total PnL (USD K)')

# size bucket
ax4 = fig2.add_subplot(gs2[1, 1])
sz_stats = trader.groupby('size_bucket', observed=True).agg(mean_pnl=('total_pnl','mean'),count=('Account','count')).reset_index()
bars4 = ax4.bar([f"{r['size_bucket']}\n(n={r['count']})" for _,r in sz_stats.iterrows()],
                sz_stats['mean_pnl']/1000, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG)
for bar, val in zip(bars4, sz_stats['mean_pnl'].values):
    ax4.text(bar.get_x()+bar.get_width()/2, (bar.get_height() if val>0 else bar.get_height())-8,
             f'${val/1000:.0f}K', ha='center', fontsize=9.5, color='white', fontweight='bold')
ax4.axhline(0, color=BORDER, lw=1)
ax4.set_title('Avg PnL by Position\nSize Segment'); ax4.set_ylabel('Mean Total PnL (USD K)')

# winners vs losers win rate by sentiment
ax5 = fig2.add_subplot(gs2[1, 2])
wl_sent = closes.merge(trader[['Account','winner']], on='Account')
wl_wr = wl_sent.groupby(['winner','simple_sent'])['is_win'].mean().unstack().reindex(columns=order)
wl_wr.index = ['Losers','Winners']
wl_wr.plot(kind='bar', ax=ax5, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG, width=0.6)
ax5.axhline(0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax5.set_xticklabels(['Losers','Winners'], rotation=0)
ax5.set_title('Win Rate by Sentiment:\nWinners vs Losers'); ax5.set_ylabel('Win Rate')
ax5.legend(['Fear/EF','Neutral','Greed/EG'], fontsize=8, framealpha=0.3)

# winners vs losers avg size by sentiment
ax6 = fig2.add_subplot(gs2[2, 0])
wl_sz = wl_sent.groupby(['winner','simple_sent'])['Size USD'].mean().unstack().reindex(columns=order)
wl_sz.index = ['Losers','Winners']
wl_sz.plot(kind='bar', ax=ax6, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG, width=0.6)
ax6.set_xticklabels(['Losers','Winners'], rotation=0)
ax6.set_title('Avg Trade Size by Sentiment:\nWinners vs Losers'); ax6.set_ylabel('Avg Size USD')
ax6.legend(['Fear/EF','Neutral','Greed/EG'], fontsize=8, framealpha=0.3)

# top 5 coin specialization
ax7 = fig2.add_subplot(gs2[2, 1:])
top5 = trader.nlargest(5,'total_pnl')['Account'].tolist()
coin_pnl = closes[closes['Account'].isin(top5)].groupby(['Account','Coin'])['Closed PnL'].sum().unstack(fill_value=0)
top_c = coin_pnl.abs().sum().nlargest(10).index
cmap = plt.get_cmap('tab10')
for i, acc in enumerate(top5):
    ax7.bar(range(len(top_c)), [coin_pnl.loc[acc, c] if c in coin_pnl.columns else 0 for c in top_c],
            bottom=[sum([coin_pnl.loc[a, c] if c in coin_pnl.columns else 0 for a in top5[:i]]) for c in top_c],
            color=cmap(i), edgecolor=BG, label=f'…{acc[-6:]}', width=0.7)
ax7.set_xticks(range(len(top_c))); ax7.set_xticklabels(top_c, rotation=30, ha='right', fontsize=8.5)
ax7.axhline(0, color='white', lw=0.8, alpha=0.5)
ax7.set_title('Top 5 Traders — PnL Breakdown by Coin'); ax7.set_ylabel('Total PnL (USD)')
ax7.legend(fontsize=8, framealpha=0.3, title='Account (last 6)')

# monthly heatmap
ax8 = fig2.add_subplot(gs2[3, :2])
pivot = closes.groupby(['Account','month'])['Closed PnL'].sum().unstack(fill_value=0)
pivot.index = [f'…{a[-6:]}' for a in pivot.index]
pivot.columns = [str(c) for c in pivot.columns]
sns.heatmap(pivot/1000, ax=ax8, cmap='RdYlGn', center=0, linewidths=0.3, linecolor=BG,
            cbar_kws={'label':'PnL (USD K)','shrink':0.5},
            annot=True, fmt='.0f', annot_kws={'size':6.5})
ax8.set_title('Monthly PnL Heatmap (USD K) — All 32 Accounts')
ax8.tick_params(axis='x', rotation=45, labelsize=7.5)
ax8.tick_params(axis='y', rotation=0, labelsize=7.5)

# Sharpe vs PnL
ax9 = fig2.add_subplot(gs2[3, 2])
ax9.scatter(trader['sharpe'].clip(-200,200), trader['total_pnl']/1000,
            c=[GREED_C if w else FEAR_C for w in trader['winner']], alpha=0.8, s=90, edgecolors='none')
ax9.axhline(0, color=BORDER, lw=1); ax9.axvline(0, color=BORDER, lw=1)
ax9.set_xlabel('Sharpe Proxy'); ax9.set_ylabel('Total PnL (USD K)')
ax9.set_title('Risk-Adjusted Performance\nvs Total PnL')

plt.savefig('/home/claude/fig2_segmentation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ fig2_segmentation.png")
