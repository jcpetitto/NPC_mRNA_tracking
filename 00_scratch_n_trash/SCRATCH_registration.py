# exact_cols = ['FoV_id', 'ne_label']
# slice_cols = [col for col in BMY1408_12_14_2023_df.columns if col.startswith('slice') and col != 'slice_id']
# all_cols = exact_cols + slice_cols
# BMY1408_12_14_2023_df.drop(columns=["slice_id"]).groupby(by=["FoV_id","ne_label"]).std()

# df_one_liner = BMY1408_12_14_2023_df.loc[:,
#     (BMY1408_12_14_2023_df.columns.isin(['FoV_id', 'ne_label'])) |
#     (BMY1408_12_14_2023_df.columns.str.startswith('slice')) &
#     (~BMY1408_12_14_2023_df.columns.isin(['slice_id']))
# ]

def calc_reg_statistics(reg_df, stats_to_keep = ['mean', 'std']):
    by_dish_reg_df = reg_df.loc[:,
    (reg_df.columns.str.startswith('FoV'))]
    by_slice_reg_df = reg_df.loc[:,
    (reg_df.columns.isin(['FoV_id', 'ne_label'])) |
    (reg_df.columns.str.startswith('slice')) &
    (~reg_df.columns.isin(['slice_id']))]

    sum_stats_by_dish_reg_df = (by_dish_reg_df
                            .describe()
                            .loc[stats_to_keep, :]
                            )

    # sum_stats_by_slice_reg_df = by_slice_reg_df.groupby(["FoV_id", "ne_label"]).describe()
    # sum_stats_by_slice_reg_df.columns.names = ['parameter','statistics']
    # sum_stats_by_slice_reg_df = sum_stats_by_slice_reg_df.loc[:, (slice(None), stats_to_keep)]
    sum_stats_by_slice_reg_df = (by_slice_reg_df
                                    .groupby(["FoV_id", "ne_label"])
                                    .describe()
                                    .rename_axis(columns=['parameter', 'statistics'])
                                    .loc[:, (slice(None), stats_to_keep)]
                                )

    return sum_stats_by_dish_reg_df, sum_stats_by_slice_reg_df


BMY1408_12_14_2023_by_dish_df, BMY1408_12_14_2023_by_slice_df = calc_reg_statistics(BMY1408_12_14_2023_df)
print(BMY1408_12_14_2023_by_dish_df)
print(BMY1408_12_14_2023_by_slice_df)

BMY1408_12_14_2023_by_dish_prec = np.linalg.norm(BMY1408_12_14_2023_by_dish_df.loc['std'].astype(np.float32))
print(BMY1408_12_14_2023_by_dish_prec)
