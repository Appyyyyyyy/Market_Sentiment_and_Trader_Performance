import pandas as pd, numpy as np, matplotlib, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')

BG='#0d1117'; PANEL='#161b22'; BORDER='#30363d'; TEXT='#c9d1d9'; MUTED='#8b949e'
FEAR_C='#f85149'; GREED_C='#3fb950'; NEUTRAL_C='#d29922'; ACCENT1='#79c0ff'
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
all_trades = merged.copy()
all_trades['is_long'] = all_trades['Direction'].isin(['Open Long','Close Short','Long > Short'])
all_trades['is_short'] = all_trades['Direction'].isin(['Open Short','Close Long','Short > Long'])

daily_closes = closes.groupby(['date','sentiment','simple_sent','fg_value']).agg(
    total_pnl=('Closed PnL','sum'), n_closes=('Closed PnL','count'),
    win_rate=('is_win','mean'), pnl_std=('Closed PnL','std'),
).reset_index()
daily_all = all_trades.groupby(['date','fg_value']).agg(
    n_trades=('Account','count'), avg_size_usd=('Size USD','mean'),
    total_volume=('Size USD','sum'), n_longs=('is_long','sum'), n_shorts=('is_short','sum'),
).reset_index()
daily_all['ls_ratio'] = daily_all['n_longs'] / (daily_all['n_longs']+daily_all['n_shorts']+1e-6)
daily = daily_closes.merge(daily_all[['date','n_trades','avg_size_usd','total_volume','ls_ratio']], on='date', how='left')
daily = daily.sort_values('date').reset_index(drop=True)

# Build features
daily['lag1_pnl']    = daily['total_pnl'].shift(1)
daily['lag2_pnl']    = daily['total_pnl'].shift(2)
daily['lag1_wr']     = daily['win_rate'].shift(1)
daily['lag1_ntrade'] = daily['n_trades'].shift(1)
daily['lag1_lsratio']= daily['ls_ratio'].shift(1)
daily['lag1_vol']    = daily['total_volume'].shift(1)
daily['fg_change']   = daily['fg_value'].diff()
daily['sent_enc']    = (daily['simple_sent']=='Greed/Extreme Greed').astype(int)
daily['pnl_bucket']  = pd.cut(daily['total_pnl'],
                               [-1e9, daily['total_pnl'].quantile(0.33), daily['total_pnl'].quantile(0.67), 1e9],
                               labels=['Low','Mid','High'])
daily = daily.dropna()

feat_cols = ['lag1_pnl','lag2_pnl','lag1_wr','lag1_ntrade','lag1_lsratio','lag1_vol',
             'fg_value','fg_change','sent_enc','n_trades','ls_ratio','avg_size_usd']
X = daily[feat_cols].fillna(0)
y = daily['pnl_bucket']
le = LabelEncoder(); y_enc = le.fit_transform(y)

rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y_enc, cv=cv, scoring='accuracy')
rf.fit(X, y_enc)
y_pred = cross_val_predict(rf, X, y_enc, cv=cv)
cm = confusion_matrix(y_enc, y_pred)

print(f"RF CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

fig3 = plt.figure(figsize=(20, 16), facecolor=BG)
gs3 = gridspec.GridSpec(2, 3, figure=fig3, hspace=0.45, wspace=0.35,
                        top=0.93, bottom=0.06, left=0.07, right=0.97)
fig3.text(0.5, 0.97, 'Predictive Model  —  Next-Day PnL Bucket Classification',
          ha='center', fontsize=16, fontweight='bold', color='#f0f6fc')
fig3.text(0.5, 0.955, f'Random Forest  |  5-Fold CV  |  Mean Accuracy = {cv_scores.mean():.1%}  |  n = {len(X)} days',
          ha='center', fontsize=10, color=MUTED)

# Feature importance
ax = fig3.add_subplot(gs3[0, 0])
imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values()
bar_cols = [GREED_C if v >= imp.median() else FEAR_C for v in imp]
ax.barh(imp.index, imp.values, color=bar_cols, edgecolor=BG)
ax.set_title('Feature Importance')
ax.set_xlabel('Importance Score')

# CV scores
ax2 = fig3.add_subplot(gs3[0, 1])
fold_labels = [f'Fold {i+1}' for i in range(5)]
bar_cv = [GREED_C if s > cv_scores.mean() else FEAR_C for s in cv_scores]
ax2.bar(fold_labels, cv_scores, color=bar_cv, edgecolor=BG)
ax2.axhline(cv_scores.mean(), color='white', lw=1.8, ls='--', label=f'Mean {cv_scores.mean():.2%}')
ax2.axhline(1/3, color=NEUTRAL_C, lw=1, ls=':', label='Random baseline 33%')
ax2.set_title('5-Fold CV Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.legend(fontsize=8)
for i, (bar, val) in enumerate(zip(ax2.patches, cv_scores)):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.1%}',
             ha='center', fontsize=9, color='white')

# Confusion matrix
ax3 = fig3.add_subplot(gs3[0, 2])
sns.heatmap(cm, ax=ax3, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_,
            linewidths=0.5, linecolor=BG, cbar_kws={'shrink':0.7})
ax3.set_title('Confusion Matrix (CV Predictions)')
ax3.set_xlabel('Predicted'); ax3.set_ylabel('Actual')

# Scatter: predicted vs actual PnL (using mean of bucket predictions)
ax4 = fig3.add_subplot(gs3[1, :2])
pred_proba = cross_val_predict(rf, X, y_enc, cv=cv, method='predict_proba')
high_prob = pred_proba[:, list(le.classes_).index('High')]
ax4.scatter(daily['date'], daily['total_pnl']/1000, c=high_prob, cmap='RdYlGn',
            alpha=0.7, s=35, vmin=0, vmax=1)
ax4.axhline(0, color=BORDER, lw=1)
ax4.set_title('Actual Daily PnL — colored by Model P(High-PnL day)')
ax4.set_ylabel('Total PnL (USD K)')
ax4.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %y'))
sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(0,1))
plt.colorbar(sm, ax=ax4, label='P(High PnL day)', shrink=0.7)

# Sentiment regime vs prediction accuracy
ax5 = fig3.add_subplot(gs3[1, 2])
daily['correct'] = (y_enc == y_pred).astype(int)
acc_by_sent = daily.groupby('simple_sent')['correct'].mean()
bars5 = ax5.bar(range(len(acc_by_sent)), acc_by_sent.values,
                color=[FEAR_C, NEUTRAL_C, GREED_C][:len(acc_by_sent)], edgecolor=BG)
for bar, val in zip(bars5, acc_by_sent.values):
    ax5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
             f'{val:.1%}', ha='center', fontsize=10, color='white', fontweight='bold')
ax5.set_xticks(range(len(acc_by_sent)))
ax5.set_xticklabels([s.replace(' ','  ') for s in acc_by_sent.index], fontsize=8, rotation=10)
ax5.set_title('Model Accuracy\nby Sentiment Regime')
ax5.set_ylabel('Accuracy')
ax5.set_ylim(0, 1)
ax5.axhline(1/3, color=NEUTRAL_C, lw=1, ls=':', label='Baseline')
ax5.legend(fontsize=8)

plt.savefig('/home/claude/fig3_model.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✓ fig3_model.png")
