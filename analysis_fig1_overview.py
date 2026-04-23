import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── STYLE ────────────────────────────────────────────────────────────────────
BG      = '#0d1117'
PANEL   = '#161b22'
BORDER  = '#30363d'
TEXT    = '#c9d1d9'
MUTED   = '#8b949e'
FEAR_C  = '#f85149'
EGREED_C= '#56d364'
GREED_C = '#3fb950'
EFEAR_C = '#ff6b6b'
NEUTRAL_C='#d29922'
ACCENT1 = '#79c0ff'
ACCENT2 = '#d2a8ff'
ACCENT3 = '#ffa657'

plt.rcParams.update({
    'figure.facecolor':BG,'axes.facecolor':PANEL,'axes.edgecolor':BORDER,
    'axes.labelcolor':TEXT,'xtick.color':MUTED,'ytick.color':MUTED,
    'text.color':TEXT,'grid.color':BORDER,'grid.alpha':0.7,
    'font.family':'DejaVu Sans','font.size':9.5,
    'axes.titlesize':11,'axes.titleweight':'bold',
    'legend.facecolor':PANEL,'legend.edgecolor':BORDER,
    'axes.spines.top':False,'axes.spines.right':False,
})

SENT_COLORS = {'Extreme Fear':EFEAR_C,'Fear':FEAR_C,'Neutral':NEUTRAL_C,
               'Greed':GREED_C,'Extreme Greed':EGREED_C}
SIMPLE_COLORS = {'Fear/Extreme Fear':FEAR_C,'Neutral':NEUTRAL_C,'Greed/Extreme Greed':GREED_C}

# ── LOAD ─────────────────────────────────────────────────────────────────────
print("Loading real data...")
trades = pd.read_csv('/mnt/user-data/uploads/historical_data__1_.csv')
fg     = pd.read_csv('/mnt/user-data/uploads/fear_greed_index__1_.csv')

# ── PART A: PREP ─────────────────────────────────────────────────────────────
print(f"\n=== PART A: DATA PREP ===")
print(f"Trades:    {trades.shape[0]:,} rows × {trades.shape[1]} cols")
print(f"Sentiment: {fg.shape[0]:,} rows × {fg.shape[1]} cols")
print(f"Trade missing: {trades.isnull().sum().sum()} | Dupes: {trades.duplicated().sum()}")
print(f"FG missing:    {fg.isnull().sum().sum()} | Dupes: {fg.duplicated().sum()}")

# Timestamps
trades['date'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True).dt.normalize()
fg['date']     = pd.to_datetime(fg['date'])

# Simplify sentiment into 3 buckets
fg['simple_sent'] = fg['classification'].map({
    'Extreme Fear':'Fear/Extreme Fear','Fear':'Fear/Extreme Fear',
    'Neutral':'Neutral',
    'Greed':'Greed/Extreme Greed','Extreme Greed':'Greed/Extreme Greed'
})

# Merge
merged = trades.merge(fg[['date','classification','value','simple_sent']], on='date', how='left')
merged.rename(columns={'classification':'sentiment','value':'fg_value'}, inplace=True)

# Only keep closings (non-zero PnL) for PnL analysis; all rows for behavior
closes = merged[merged['Closed PnL'] != 0].copy()
closes['is_win'] = closes['Closed PnL'] > 0
closes['net_pnl'] = closes['Closed PnL'] - closes['Fee']

all_trades = merged.copy()
all_trades['is_long'] = all_trades['Direction'].isin(['Open Long','Close Short','Long > Short'])
all_trades['is_short'] = all_trades['Direction'].isin(['Open Short','Close Long','Short > Long'])

print(f"\nTotal trades:   {len(merged):,}")
print(f"Closed trades (non-zero PnL): {len(closes):,}")
print(f"Accounts: {merged['Account'].nunique()}")
print(f"Date range: {merged['date'].min().date()} → {merged['date'].max().date()}")

# ── DAILY METRICS ────────────────────────────────────────────────────────────
daily_closes = closes.groupby(['date','sentiment','simple_sent','fg_value']).agg(
    total_pnl     = ('Closed PnL','sum'),
    net_pnl       = ('net_pnl','sum'),
    n_closes      = ('Closed PnL','count'),
    win_rate      = ('is_win','mean'),
    avg_pnl_trade = ('Closed PnL','mean'),
    pnl_std       = ('Closed PnL','std'),
).reset_index()

daily_all = all_trades.groupby(['date','sentiment','simple_sent','fg_value']).agg(
    n_trades      = ('Account','count'),
    n_traders     = ('Account','nunique'),
    avg_size_usd  = ('Size USD','mean'),
    total_volume  = ('Size USD','sum'),
    n_longs       = ('is_long','sum'),
    n_shorts      = ('is_short','sum'),
    n_liq         = ('Direction', lambda x: x.isin(['Liquidated Isolated Short','Auto-Deleveraging']).sum()),
).reset_index()
daily_all['ls_ratio'] = daily_all['n_longs'] / (daily_all['n_longs'] + daily_all['n_shorts'] + 1e-6)

daily = daily_closes.merge(daily_all, on=['date','sentiment','simple_sent','fg_value'], how='outer')
daily = daily.sort_values('date').reset_index(drop=True)

# ── TRADER-LEVEL METRICS ─────────────────────────────────────────────────────
trader = closes.groupby('Account').agg(
    total_pnl   = ('Closed PnL','sum'),
    net_pnl     = ('net_pnl','sum'),
    n_trades    = ('Closed PnL','count'),
    win_rate    = ('is_win','mean'),
    avg_size    = ('Size USD','mean'),
    pnl_std     = ('Closed PnL','std'),
    total_fees  = ('Fee','sum'),
).reset_index()
trader['avg_pnl_per_trade'] = trader['total_pnl'] / trader['n_trades']
trader['sharpe'] = trader['total_pnl'] / (trader['pnl_std'] + 1e-6)
trader['freq_bucket'] = pd.qcut(trader['n_trades'],3,labels=['Infrequent','Medium','Frequent'])
trader['size_bucket']  = pd.qcut(trader['avg_size'],3,labels=['Small','Medium','Large'])
trader['winner']  = trader['total_pnl'] > 0

# Sentiment-split trader behaviour
trader_sent = closes.groupby(['Account','simple_sent']).agg(
    total_pnl = ('Closed PnL','sum'),
    n_trades  = ('Closed PnL','count'),
    win_rate  = ('is_win','mean'),
    avg_size  = ('Size USD','mean'),
).reset_index()

print("\n=== KEY STATS ===")
for s in ['Fear/Extreme Fear','Neutral','Greed/Extreme Greed']:
    d = daily[daily['simple_sent']==s]
    print(f"{s}: {len(d)} days | median PnL ${d['total_pnl'].median():,.0f} | win rate {d['win_rate'].mean():.1%}")

fear_days  = daily[daily['simple_sent']=='Fear/Extreme Fear']
greed_days = daily[daily['simple_sent']=='Greed/Extreme Greed']
t_stat, p_val = stats.ttest_ind(fear_days['total_pnl'].dropna(), greed_days['total_pnl'].dropna())
print(f"\nT-test (Fear vs Greed PnL): t={t_stat:.2f}, p={p_val:.4f}")

print("\nTop 5 accounts by total PnL:")
print(trader.nlargest(5,'total_pnl')[['Account','total_pnl','net_pnl','n_trades','win_rate']].to_string())

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — OVERVIEW DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 24), facecolor=BG)
gs  = gridspec.GridSpec(5, 3, figure=fig, hspace=0.45, wspace=0.35,
                        top=0.95, bottom=0.04, left=0.07, right=0.97)

fig.text(0.5, 0.975, 'Hyperliquid × Fear/Greed Index  —  Sentiment-Behavior Analysis',
         ha='center', fontsize=18, fontweight='bold', color='#f0f6fc')
fig.text(0.5, 0.961, f'Jan 2024 – Dec 2024  |  {merged["Account"].nunique()} Accounts  |  {len(merged):,} Trades  |  {len(closes):,} Closing Events',
         ha='center', fontsize=10.5, color=MUTED)

# Row 0 — Timeline PnL
ax0 = fig.add_subplot(gs[0, :2])
for s, col in [('Fear/Extreme Fear',FEAR_C),('Neutral',NEUTRAL_C),('Greed/Extreme Greed',GREED_C)]:
    d = daily[daily['simple_sent']==s].sort_values('date')
    ax0.bar(d['date'], d['total_pnl'], color=col, alpha=0.45, width=1.0, label=s)
# rolling avg all
d_all = daily.sort_values('date')
roll = d_all.set_index('date')['total_pnl'].rolling('14D').mean()
ax0.plot(roll.index, roll.values, color='white', lw=1.8, label='14d MA', zorder=5)
ax0.axhline(0, color=BORDER, lw=1)
ax0.set_title('Daily Total Closed PnL by Sentiment Regime')
ax0.set_ylabel('Total Closed PnL (USD)')
ax0.legend(fontsize=8, ncol=4, framealpha=0.3, loc='upper left')
ax0.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

# Fear/Greed value timeline (secondary)
ax0b = ax0.twinx()
d_all2 = daily.sort_values('date').dropna(subset=['fg_value'])
ax0b.plot(d_all2['date'], d_all2['fg_value'], color=ACCENT1, lw=0.8, alpha=0.5)
ax0b.set_ylabel('F/G Index', color=ACCENT1)
ax0b.tick_params(axis='y', colors=ACCENT1)
ax0b.set_ylim(0,100)

# Row 0 col 2 — Sentiment day counts
ax1 = fig.add_subplot(gs[0, 2])
fg_overlap = fg[(fg['date'] >= merged['date'].min()) & (fg['date'] <= merged['date'].max())]
counts = fg_overlap['classification'].value_counts().reindex(
    ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed'], fill_value=0)
colors_bar = [SENT_COLORS[c] for c in counts.index]
bars = ax1.bar(range(len(counts)), counts.values, color=colors_bar, edgecolor=BG, lw=1.2)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5, str(val),
             ha='center', fontsize=8.5, color=TEXT)
ax1.set_xticks(range(len(counts)))
ax1.set_xticklabels(['Ext.\nFear','Fear','Neutral','Greed','Ext.\nGreed'], fontsize=8)
ax1.set_title('Trading Days\nby Sentiment Class')
ax1.set_ylabel('# Days')

# Row 1 — Boxplot PnL per close
ax2 = fig.add_subplot(gs[1, 0])
order = ['Fear/Extreme Fear','Neutral','Greed/Extreme Greed']
data_box = [closes[closes['simple_sent']==s]['Closed PnL'].clip(-3000,3000).values for s in order]
bp = ax2.boxplot(data_box, patch_artist=True,
                 labels=['Fear/\nExt Fear','Neutral','Greed/\nExt Greed'],
                 medianprops=dict(color='white',lw=2.5),
                 whiskerprops=dict(color=MUTED,lw=1.2),
                 capprops=dict(color=MUTED),
                 flierprops=dict(marker='.',markersize=1.5,alpha=0.15,color=MUTED))
for patch, col in zip(bp['boxes'], [FEAR_C, NEUTRAL_C, GREED_C]):
    patch.set_facecolor(col); patch.set_alpha(0.55)
ax2.set_title('PnL per Closing Trade\n(clipped ±$3K)')
ax2.set_ylabel('Closed PnL (USD)')
ax2.axhline(0, color=BORDER, lw=1)

# Row 1 — Win Rate
ax3 = fig.add_subplot(gs[1, 1])
wr = daily.groupby('simple_sent')['win_rate'].agg(['mean','sem']).reindex(order)
xpos = range(len(order))
bars3 = ax3.bar(xpos, wr['mean']*100, yerr=wr['sem']*196,
                color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG,
                capsize=6, error_kw=dict(color=MUTED,lw=1.5))
for bar, val in zip(bars3, wr['mean'].values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
             f'{val:.1%}', ha='center', fontsize=9.5, color='white', fontweight='bold')
ax3.axhline(50, color='white', lw=0.8, ls='--', alpha=0.4)
ax3.set_xticks(xpos); ax3.set_xticklabels(['Fear/\nExt Fear','Neutral','Greed/\nExt Greed'], fontsize=8.5)
ax3.set_title('Win Rate by Sentiment')
ax3.set_ylabel('Win Rate (%)')
ax3.set_ylim(0, 80)

# Row 1 — Avg Trade Size
ax4 = fig.add_subplot(gs[1, 2])
sz = daily.groupby('simple_sent')['avg_size_usd'].agg(['mean','sem']).reindex(order)
bars4 = ax4.bar(xpos, sz['mean'], yerr=sz['sem']*1.96,
                color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG,
                capsize=6, error_kw=dict(color=MUTED,lw=1.5))
for bar, val in zip(bars4, sz['mean'].values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
             f'${val:,.0f}', ha='center', fontsize=8.5, color='white')
ax4.set_xticks(xpos); ax4.set_xticklabels(['Fear/\nExt Fear','Neutral','Greed/\nExt Greed'], fontsize=8.5)
ax4.set_title('Avg Trade Size (USD)')
ax4.set_ylabel('Avg Size USD')

# Row 2 — Trade frequency
ax5 = fig.add_subplot(gs[2, 0])
tf = daily.groupby('simple_sent')['n_trades'].agg(['mean','sem']).reindex(order)
bars5 = ax5.bar(xpos, tf['mean'], yerr=tf['sem']*1.96,
                color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG,
                capsize=6, error_kw=dict(color=MUTED,lw=1.5))
for bar, val in zip(bars5, tf['mean'].values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val:.0f}', ha='center', fontsize=9.5, color='white', fontweight='bold')
ax5.set_xticks(xpos); ax5.set_xticklabels(['Fear/\nExt Fear','Neutral','Greed/\nExt Greed'], fontsize=8.5)
ax5.set_title('Avg Daily Trade Count')
ax5.set_ylabel('# Trades / Day')

# Row 2 — Long/Short ratio
ax6 = fig.add_subplot(gs[2, 1])
ls = daily.groupby('simple_sent')['ls_ratio'].mean().reindex(order)
bars6 = ax6.bar(xpos, ls.values*100, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG)
ax6.axhline(50, color='white', lw=1, ls='--', alpha=0.5, label='50% line')
for bar, val in zip(bars6, ls.values):
    ax6.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val:.1%}', ha='center', fontsize=9.5, color='white', fontweight='bold')
ax6.set_xticks(xpos); ax6.set_xticklabels(['Fear/\nExt Fear','Neutral','Greed/\nExt Greed'], fontsize=8.5)
ax6.set_title('Long Ratio (% of Directional Trades)')
ax6.set_ylabel('Long %')
ax6.set_ylim(0, 80)
ax6.legend(fontsize=8)

# Row 2 — Total volume
ax7 = fig.add_subplot(gs[2, 2])
vol = daily.groupby('simple_sent')['total_volume'].agg(['mean','sem']).reindex(order)
bars7 = ax7.bar(xpos, vol['mean']/1e6, yerr=vol['sem']*1.96/1e6,
                color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG,
                capsize=6, error_kw=dict(color=MUTED,lw=1.5))
for bar, val in zip(bars7, vol['mean'].values):
    ax7.text(bar.get_x()+bar.get_width()/2, bar.get_height()/1e6+0.01,
             f'${val/1e6:.2f}M', ha='center', fontsize=8.5, color='white')
ax7.set_xticks(xpos); ax7.set_xticklabels(['Fear/\nExt Fear','Neutral','Greed/\nExt Greed'], fontsize=8.5)
ax7.set_title('Avg Daily Volume (USD)')
ax7.set_ylabel('Volume (USD M)')

# Row 3 — FG value vs win rate scatter
ax8 = fig.add_subplot(gs[3, 0])
d_sc = daily.dropna(subset=['fg_value','win_rate'])
scatter_colors = [SENT_COLORS.get(s,'grey') for s in d_sc['sentiment']]
ax8.scatter(d_sc['fg_value'], d_sc['win_rate']*100, c=scatter_colors, alpha=0.55, s=30)
# regression line
m,b,r,p,se = stats.linregress(d_sc['fg_value'], d_sc['win_rate']*100)
xr = np.linspace(0,100,100)
ax8.plot(xr, m*xr+b, color='white', lw=1.5, ls='--', label=f'r={r:.2f}, p={p:.3f}')
ax8.set_xlabel('Fear/Greed Index Value')
ax8.set_ylabel('Win Rate (%)')
ax8.set_title('F/G Index Value vs Daily Win Rate')
ax8.legend(fontsize=8)

# Row 3 — FG value vs total PnL
ax9 = fig.add_subplot(gs[3, 1])
d_sc2 = daily.dropna(subset=['fg_value','total_pnl'])
sc_cols2 = [SENT_COLORS.get(s,'grey') for s in d_sc2['sentiment']]
ax9.scatter(d_sc2['fg_value'], d_sc2['total_pnl'], c=sc_cols2, alpha=0.45, s=30)
m2,b2,r2,p2,_ = stats.linregress(d_sc2['fg_value'], d_sc2['total_pnl'])
ax9.plot(xr, m2*xr+b2, color='white', lw=1.5, ls='--', label=f'r={r2:.2f}, p={p2:.3f}')
ax9.axhline(0, color=BORDER, lw=1)
ax9.set_xlabel('Fear/Greed Index Value')
ax9.set_ylabel('Total PnL (USD)')
ax9.set_title('F/G Index Value vs Daily Total PnL')
ax9.legend(fontsize=8)

# Row 3 — Heatmap PnL by weekday × sentiment
ax10 = fig.add_subplot(gs[3, 2])
daily['weekday'] = pd.to_datetime(daily['date']).dt.day_name()
week_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
heat = daily.groupby(['weekday','simple_sent'])['total_pnl'].mean().unstack()
heat = heat.reindex(week_order)
sns.heatmap(heat.T, ax=ax10, cmap='RdYlGn', center=0, annot=True, fmt='.0f',
            linewidths=0.5, linecolor=BG, cbar_kws={'shrink':0.6,'label':'Avg PnL'})
ax10.set_title('Avg PnL: Weekday × Sentiment')
ax10.set_xlabel('')

# Row 4 — Coin performance by sentiment
ax11 = fig.add_subplot(gs[4, :2])
top_coins = closes.groupby('Coin')['Closed PnL'].sum().abs().nlargest(12).index
coin_sent = closes[closes['Coin'].isin(top_coins)].groupby(['Coin','simple_sent'])['Closed PnL'].sum().unstack(fill_value=0)
coin_sent = coin_sent.reindex(columns=order, fill_value=0)
coin_sent = coin_sent.reindex(coin_sent.sum(axis=1).sort_values().index)
coin_sent.plot(kind='barh', ax=ax11, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG, width=0.65)
ax11.axvline(0, color='white', lw=0.8, alpha=0.5)
ax11.set_title('Total PnL by Top 12 Coins × Sentiment')
ax11.set_xlabel('Total Closed PnL (USD)')
ax11.legend(['Fear/Ext Fear','Neutral','Greed/Ext Greed'], fontsize=8, framealpha=0.3)

# Row 4 — Cumulative PnL all traders
ax12 = fig.add_subplot(gs[4, 2])
top_traders = trader.nlargest(5,'total_pnl')['Account'].tolist()
bot_traders = trader.nsmallest(5,'total_pnl')['Account'].tolist()
for acc in top_traders:
    d = closes[closes['Account']==acc].sort_values('date')
    ax12.plot(d['date'], d['Closed PnL'].cumsum()/1000, color=GREED_C, alpha=0.7, lw=1.3)
for acc in bot_traders:
    d = closes[closes['Account']==acc].sort_values('date')
    ax12.plot(d['date'], d['Closed PnL'].cumsum()/1000, color=FEAR_C, alpha=0.7, lw=1.3)
ax12.axhline(0, color=BORDER, lw=1)
ax12.set_title('Cumulative PnL\nTop 5 (green) vs Bottom 5 (red)')
ax12.set_ylabel('Cum PnL (USD K)')
ax12.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))

plt.savefig('/home/claude/fig1_overview_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ fig1_overview_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — TRADER SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
import pandas as pd, numpy as np, matplotlib, matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec, matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# reload vars since this is appended
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
trader = closes.groupby('Account').agg(
    total_pnl=('Closed PnL','sum'),net_pnl=('net_pnl','sum'),
    n_trades=('Closed PnL','count'),win_rate=('is_win','mean'),
    avg_size=('Size USD','mean'),pnl_std=('Closed PnL','std'),total_fees=('Fee','sum'),
).reset_index()
trader['avg_pnl_per_trade'] = trader['total_pnl'] / trader['n_trades']
trader['sharpe'] = trader['total_pnl'] / (trader['pnl_std'] + 1e-6)
trader['freq_bucket'] = pd.qcut(trader['n_trades'],3,labels=['Infrequent','Medium','Frequent'])
trader['size_bucket']  = pd.qcut(trader['avg_size'],3,labels=['Small','Medium','Large'])
trader['winner'] = trader['total_pnl'] > 0
FEAR_C='#f85149'; GREED_C='#3fb950'; NEUTRAL_C='#d29922'
EGREED_C='#56d364'; EFEAR_C='#ff6b6b'; ACCENT1='#79c0ff'; ACCENT2='#d2a8ff'; ACCENT3='#ffa657'
BG='#0d1117'; PANEL='#161b22'; BORDER='#30363d'; TEXT='#c9d1d9'; MUTED='#8b949e'
order = ['Fear/Extreme Fear','Neutral','Greed/Extreme Greed']

fig2 = plt.figure(figsize=(20, 22), facecolor=BG)
gs2 = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.45, wspace=0.35,
                        top=0.95, bottom=0.04, left=0.07, right=0.97)
fig2.text(0.5, 0.975, 'Trader Segmentation  —  Who Wins, Who Loses, and Why',
          ha='center', fontsize=17, fontweight='bold', color='#f0f6fc')

# Panel 1 — scatter total PnL vs win rate, bubble=n_trades
ax = fig2.add_subplot(gs2[0, :2])
scatter_c = [GREED_C if w else FEAR_C for w in trader['winner']]
sizes = np.clip(trader['n_trades']/50, 20, 400)
sc = ax.scatter(trader['win_rate']*100, trader['total_pnl']/1000,
                s=sizes, c=scatter_c, alpha=0.75, edgecolors='none')
for _, row in trader.nlargest(5,'total_pnl').iterrows():
    ax.annotate(row['Account'][-6:], (row['win_rate']*100, row['total_pnl']/1000),
                fontsize=7.5, color=TEXT, xytext=(5,3), textcoords='offset points')
ax.axhline(0, color=BORDER, lw=1); ax.axvline(50, color=BORDER, lw=1)
ax.set_xlabel('Win Rate (%)'); ax.set_ylabel('Total PnL (USD K)')
ax.set_title('Win Rate vs Total PnL  (bubble size = trade count  |  green=winner  red=loser)')

# Panel 2 — trader ranking bar
ax2b = fig2.add_subplot(gs2[0, 2])
tr_sorted = trader.sort_values('total_pnl')
colors_rank = [GREED_C if p > 0 else FEAR_C for p in tr_sorted['total_pnl']]
ax2b.barh(range(len(tr_sorted)), tr_sorted['total_pnl']/1000, color=colors_rank, edgecolor=BG, height=0.8)
ax2b.axvline(0, color='white', lw=0.8)
ax2b.set_xlabel('Total PnL (USD K)')
ax2b.set_title(f'All 32 Traders Ranked\nby Total PnL')
ax2b.set_yticks([])

# Panel 3 — Freq bucket: PnL and win rate
ax3 = fig2.add_subplot(gs2[1, 0])
freq_stats = trader.groupby('freq_bucket', observed=True).agg(
    mean_pnl=('total_pnl','mean'), mean_wr=('win_rate','mean'),
    count=('Account','count')).reset_index()
freq_stats['label'] = freq_stats.apply(lambda r: f"{r['freq_bucket']}\n(n={r['count']})", axis=1)
bars = ax3.bar(freq_stats['label'], freq_stats['mean_pnl']/1000,
               color=[ACCENT1,ACCENT2,ACCENT3], edgecolor=BG)
for bar, val in zip(bars, freq_stats['mean_pnl'].values):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2 if val>0 else bar.get_height()-15,
             f'${val/1000:.0f}K', ha='center', fontsize=9, color='white', fontweight='bold')
ax3.axhline(0, color=BORDER, lw=1)
ax3.set_title('Avg PnL by Trade Frequency\nSegment')
ax3.set_ylabel('Mean Total PnL (USD K)')

# Panel 4 — size bucket
ax4 = fig2.add_subplot(gs2[1, 1])
sz_stats = trader.groupby('size_bucket', observed=True).agg(
    mean_pnl=('total_pnl','mean'), mean_wr=('win_rate','mean'),
    count=('Account','count')).reset_index()
sz_stats['label'] = sz_stats.apply(lambda r: f"{r['size_bucket']}\n(n={r['count']})", axis=1)
bars4 = ax4.bar(sz_stats['label'], sz_stats['mean_pnl']/1000,
                color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG)
for bar, val in zip(bars4, sz_stats['mean_pnl'].values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2 if val>0 else bar.get_height()-15,
             f'${val/1000:.0f}K', ha='center', fontsize=9, color='white', fontweight='bold')
ax4.axhline(0, color=BORDER, lw=1)
ax4.set_title('Avg PnL by Position Size\nSegment')
ax4.set_ylabel('Mean Total PnL (USD K)')

# Panel 5 — Winners vs Losers win rate by sentiment
ax5 = fig2.add_subplot(gs2[1, 2])
wl_sent = closes.merge(trader[['Account','winner']], on='Account')
wl_wr = wl_sent.groupby(['winner','simple_sent'])['is_win'].mean().unstack().reindex(columns=order)
wl_wr.index = ['Losers','Winners']
wl_wr.plot(kind='bar', ax=ax5, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG, width=0.6)
ax5.axhline(0.5, color='white', lw=0.8, ls='--', alpha=0.5)
ax5.set_xticklabels(['Losers','Winners'], rotation=0)
ax5.set_title('Win Rate by Sentiment:\nWinners vs Losers')
ax5.set_ylabel('Win Rate')
ax5.legend(['Fear/EF','Neutral','Greed/EG'], fontsize=7.5, framealpha=0.3)

# Panel 6 — Behaviour change on Fear vs Greed days: avg size
ax6 = fig2.add_subplot(gs2[2, 0])
wl_sz = wl_sent.groupby(['winner','simple_sent'])['Size USD'].mean().unstack().reindex(columns=order)
wl_sz.index = ['Losers','Winners']
wl_sz.plot(kind='bar', ax=ax6, color=[FEAR_C,NEUTRAL_C,GREED_C], edgecolor=BG, width=0.6)
ax6.set_xticklabels(['Losers','Winners'], rotation=0)
ax6.set_title('Avg Trade Size by Sentiment:\nWinners vs Losers')
ax6.set_ylabel('Avg Size USD')
ax6.legend(['Fear/EF','Neutral','Greed/EG'], fontsize=7.5, framealpha=0.3)

# Panel 7 — Coin specialization: top 5 accounts
ax7 = fig2.add_subplot(gs2[2, 1:])
top5 = trader.nlargest(5,'total_pnl')['Account'].tolist()
coin_pnl = closes[closes['Account'].isin(top5)].groupby(['Account','Coin'])['Closed PnL'].sum()
coin_pnl = coin_pnl.unstack(fill_value=0)
top_coins_by_abs = coin_pnl.abs().sum().nlargest(10).index
coin_pnl[top_coins_by_abs].T.plot(kind='bar', ax=ax7, edgecolor=BG, colormap='tab10', width=0.75)
ax7.axhline(0, color='white', lw=0.8, alpha=0.5)
ax7.set_title('Top 5 Traders — PnL by Coin')
ax7.set_ylabel('Total PnL (USD)')
ax7.set_xlabel('Coin')
ax7.legend([a[-6:] for a in top5], fontsize=7.5, framealpha=0.3, title='Acct (last 6)')
ax7.tick_params(axis='x', rotation=30)

# Panel 8 — monthly PnL heatmap
ax8 = fig2.add_subplot(gs2[3, :2])
closes['month'] = closes['date'].dt.to_period('M')
monthly = closes.groupby('month')['Closed PnL'].sum().reset_index()
monthly['month_str'] = monthly['month'].astype(str)
closes_m = closes.merge(monthly[['month','Closed PnL']].rename(columns={'Closed PnL':'m_pnl'}), on='month')
# Pivot: account × month
pivot = closes.groupby(['Account','month'])['Closed PnL'].sum().unstack(fill_value=0)
pivot.index = [a[-6:] for a in pivot.index]
pivot.columns = [str(c) for c in pivot.columns]
sns.heatmap(pivot, ax=ax8, cmap='RdYlGn', center=0, linewidths=0.2, linecolor=BG,
            cbar_kws={'label':'Monthly PnL','shrink':0.5}, fmt='.0f',
            annot=len(pivot.columns) <= 18)
ax8.set_title('Monthly PnL Heatmap — All Accounts (last 6 chars of address)')
ax8.set_xlabel('Month')
ax8.tick_params(axis='x', rotation=45, labelsize=7)
ax8.tick_params(axis='y', rotation=0, labelsize=7.5)

# Panel 9 — Sharpe vs total PnL
ax9 = fig2.add_subplot(gs2[3, 2])
sc_colors9 = [GREED_C if w else FEAR_C for w in trader['winner']]
ax9.scatter(trader['sharpe'].clip(-200,200), trader['total_pnl']/1000,
            c=sc_colors9, alpha=0.75, s=80, edgecolors='none')
ax9.axhline(0, color=BORDER, lw=1); ax9.axvline(0, color=BORDER, lw=1)
ax9.set_xlabel('Sharpe Proxy (PnL / PnL Std)')
ax9.set_ylabel('Total PnL (USD K)')
ax9.set_title('Risk-Adjusted Performance\nvs Total PnL')

plt.savefig('/home/claude/fig2_segmentation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ fig2_segmentation.png")
