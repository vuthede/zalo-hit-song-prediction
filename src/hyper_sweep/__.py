

def calc_result(x):

    huge_df = DataFrame(random.randn(100000, 5), columns=['A', 'B', 'C', 'D', 'E'])

    total = 0

    # Assume that I MUST iterate
    for idx_and_row in huge_df.iterrows():
        idx = idx_and_row[0]
        row = idx_and_row[1]


        # Assume there is no way to optimize here
        curr_sum = row['A'] * x['adjustment_1'] + \
                   row['B'] * x['adjustment_2'] + \
                   row['C'] * x['adjustment_3'] + \
                   row['D'] * x['adjustment_4'] + \
                   row['E'] * x['adjustment_5']


        total += curr_sum

    # In real life I want the total as high as possible, but for the minimizer, it has to negative a negative value
    total_as_neg = total * -1

    print(total_as_neg)

    return total_as_neg



def calc_result(x, reporter):  # add a reporter param here

    huge_df = DataFrame(random.randn(100000, 5), columns=['A', 'B', 'C', 'D', 'E'])

    total = 0

    # Assume that I MUST iterate
    for idx_and_row in huge_df.iterrows():
        idx = idx_and_row[0]
        row = idx_and_row[1]


        # Assume there is no way to optimize here
        curr_sum = row['A'] * x['adjustment_1'] + \
                   row['B'] * x['adjustment_2'] + \
                   row['C'] * x['adjustment_3'] + \
                   row['D'] * x['adjustment_4'] + \
                   row['E'] * x['adjustment_5']


        total += curr_sum

    # In real life I want the total as high as possible, but for the minimizer, it has to negative a negative value
    # total_as_neg = total * -1

    # print(total_as_neg)

    # Ray will negate this by itself to feed into HyperOpt
    reporter(timesteps_total=1, episode_reward_mean=total)

    return total_as_neg
