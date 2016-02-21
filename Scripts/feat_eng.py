# feature engineering

## data leakage?
df_sev_all = df_sev.merge(df_all, on='id')
loc = 'location 1'
time = []
t = 0
for i in range(df_sev_all.shape[0]):
  if df_sev_all['location'][i]==loc:
    t += 1
    time += [t]
  else:
    loc = df_sev_all['location'][i]
    t = 1
    time += [t]

df_time = pd.DataFrame({'id': df_sev['id'], 'time': time})

## feature engineering
# df_loc
# contain information
loc_diff = np.setdiff1d(df_test['location'].values, df_train['location'].values)
df_loc_table = pd.pivot_table(df_all, index='id', columns='location', aggfunc=len,
  fill_value=0)
df_loc_table.drop(loc_diff, axis=1, inplace=True)

loc_freq = df_all['location'].value_counts()
loc_freq = pd.DataFrame({'loc_freq': loc_freq})
loc_freq.index.name = 'location'
df_all = df_all.join(loc_freq, on='location')

# df_all
myfun = lambda x: int(x.strip('location '))
df_all['location'] = df_all['location'].apply(myfun)

# df_eve
df_eve_table = pd.pivot_table(df_eve, index='id', columns='event_type', aggfunc=len,
	fill_value=0)
grouped = df_eve[['id', 'event_type']].groupby('id')
# df_eve_num is not useful!
myfun = lambda x: len(np.unique(x))
df_eve_num = grouped.aggregate(myfun)
myfun = lambda x: max(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_max = grouped.aggregate(myfun)
df_eve_max.rename(columns={'event_type': 'eve_max'}, inplace=True)
myfun = lambda x: min(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_min = grouped.aggregate(myfun)
df_eve_min.rename(columns={'event_type': 'eve_min'}, inplace=True)
myfun = lambda x: np.std(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_std = grouped.aggregate(myfun)
df_eve_std.rename(columns={'event_type': 'eve_std'}, inplace=True)
myfun = lambda x: skew(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_eve_skew = grouped.aggregate(myfun)
df_eve_skew.rename(columns={'event_type': 'eve_skew'}, inplace=True)

## df_log
df_log_table = pd.pivot_table(df_log, values='volume', index='id', columns='log_feature', aggfunc=np.sum,
	fill_value=0)
df_log_vol_sum = pd.DataFrame(data={'log_vol_sum': df_log_table.sum(axis=1)})
grouped = df_log[['id', 'log_feature']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_log_feat_num = grouped.aggregate(myfun)
df_log_feat_num.rename(columns={'log_feature': 'log_feat_num'}, inplace=True)
myfun = lambda x: max(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_max = grouped.aggregate(myfun)
df_log_feat_max.rename(columns={'log_feature': 'log_feat_max'}, inplace=True)
myfun = lambda x: min(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_min = grouped.aggregate(myfun)
df_log_feat_min.rename(columns={'log_feature': 'log_feat_min'}, inplace=True)
myfun = lambda x: np.std(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_std = grouped.aggregate(myfun)
df_log_feat_std.rename(columns={'log_feature': 'log_feat_std'}, inplace=True)
myfun = lambda x: skew(x.apply(lambda x: x.strip('feature ')).astype(int))
df_log_feat_skew = grouped.aggregate(myfun)
df_log_feat_skew.rename(columns={'log_feature': 'log_feat_skew'}, inplace=True)

grouped = df_log[['id', 'volume']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_log_vol_num = grouped.aggregate(myfun)
df_log_vol_num.rename(columns={'volume': 'vol_num'}, inplace=True)
# df_log_vol_max is controlled by sum and num
myfun = lambda x: min(x)
df_log_vol_min = grouped.aggregate(myfun)
df_log_vol_min.rename(columns={'volume': 'vol_min'}, inplace=True)

grouped = df_log[['log_feature', 'volume']].groupby('log_feature')
myfun = lambda x: np.sum(x)
log_feat_freq = grouped.aggregate(myfun)
log_feat_freq.rename(columns={'volume': 'log_feat_freq'}, inplace=True)
df_log_feat_freq = df_log.join(log_feat_freq, on='log_feature')
grouped = df_log_feat_freq[['id', 'log_feat_freq']].groupby('id')
myfun = lambda x: max(x)
df_log_feat_freq_max = grouped.aggregate(myfun)
df_log_feat_freq_max.rename(columns={'log_feat_freq': 'log_feat_freq_max'}, inplace=True)
myfun = lambda x: min(x)
df_log_feat_freq_min = grouped.aggregate(myfun)
df_log_feat_freq_min.rename(columns={'log_feat_freq': 'log_feat_freq_min'}, inplace=True)

# df_res
df_res_table = pd.pivot_table(df_res, index='id', columns='resource_type', aggfunc=len,
	fill_value=0)
grouped = df_res[['id', 'resource_type']].groupby('id')
myfun = lambda x: len(np.unique(x))
df_res_num = grouped.aggregate(myfun)
df_res_num.rename(columns={'resource_type': 'res_num'}, inplace=True)
myfun = lambda x: np.std(x.apply(lambda x: x.strip('resource_type ')).astype(int))
df_res_std = grouped.aggregate(myfun)
df_res_std.rename(columns={'resource_type': 'res_std'}, inplace=True)

# df_sev
# It is categorical. It does not have an ordering.
df_sev_table = pd.pivot_table(df_sev, index='id', columns='severity_type', aggfunc=len,
	fill_value=0)

# interaction
df_log_loc = df_log.merge(df_all[['id', 'location']], on='id')
grouped = df_log_loc[['volume', 'location']].groupby('location')
myfun = lambda x: np.sum(x)
df_loc_log_vol_sum = grouped.aggregate(myfun)
df_loc_log_vol_sum.rename(columns={'volume': 'loc_log_vol_sum'}, inplace=True)

grouped = df_log_loc[['log_feature', 'location']].groupby('location')
myfun = lambda x: len(np.unique(x))
df_loc_log_feat_num = grouped.aggregate(myfun)
df_loc_log_feat_num.rename(columns={'log_feature': 'loc_log_feat_num'}, inplace=True)
myfun = lambda x: max(x.apply(lambda x: x.strip('feature ')).astype(int))
df_loc_log_feat_max = grouped.aggregate(myfun)
df_loc_log_feat_max.rename(columns={'log_feature': 'loc_log_feat_max'}, inplace=True)

df_eve_loc = df_eve.merge(df_all[['id', 'location']], on='id')
grouped = df_eve_loc[['event_type', 'location']].groupby('location')
myfun = lambda x: max(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
df_loc_eve_max = grouped.aggregate(myfun)
df_loc_eve_max.rename(columns={'event_type': 'loc_eve_max'}, inplace=True)

df_res_loc = df_res.merge(df_all[['id', 'location']], on='id')
grouped = df_res_loc[['resource_type', 'location']].groupby('location')
myfun = lambda x: len(np.unique(x))
df_loc_res_num = grouped.aggregate(myfun)
df_loc_res_num.rename(columns={'resource_type': 'loc_res_num'}, inplace=True)

df_lfn_loc = df_all[['id', 'location']].join(df_log_feat_num, on='id')
grouped = df_lfn_loc[['location', 'log_feat_num']].groupby('location')
myfun = lambda x: len(np.unique(x))
df_loc_lfn_num = grouped.aggregate(myfun)
df_loc_lfn_num.rename(columns={'log_feat_num': 'loc_lfn_num'}, inplace=True)

df_loc_lfm = df_all[['id', 'location']].join(df_log_feat_max, on='id')
df_loc_lfm_freq = pd.DataFrame(data={'loc_lfm_freq': df_loc_lfm.groupby(['location', 'log_feat_max']).size()})
df_loc_lfm_freq = df_loc_lfm.join(df_loc_lfm_freq, on=['location', 'log_feat_max'])[['id', 'loc_lfm_freq']]

df_loc_lfn = df_all[['id', 'location']].join(df_log_feat_num, on='id')
df_loc_lfn_freq = pd.DataFrame(data={'loc_lfn_freq': df_loc_lfn.groupby(['location', 'log_feat_num']).size()})
df_loc_lfn_freq = df_loc_lfn.join(df_loc_lfn_freq, on=['location', 'log_feat_num'])[['id', 'loc_lfn_freq']]
