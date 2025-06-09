import numpy as np
import scipy.stats as stats

# mla dint is better than just dint
# mha is the worst
# cog_attn 

# no stat significant difference between mla and mla_dint (or mla_dint w/o bias)
def welch_t_test(a, b):
    t_value, p_value = stats.ttest_ind(a, b, equal_var=False)
    print(f"Welch's T statistic: {t_value}")
    print(f"Welch's T statistic p-value: {p_value}")
    return
# compare against some base (currently base_small)
# TODO look into automatically getting results from log files
# paired by seeds for paired tests
# mann-whitney u and wilcoxon signed rank tests

# best achived val_loss_unseen
base_small = np.array([0.02011, 0.01981, 0.01973, 0.01982, 0.01976, 0.02, 0.01968, 0.01957])
# base_mha = np.array([0.02055, 0.02062, 0.02037])
# base_mla_dint = np.array([0.01945, 0.01941, 0.01976])
# dint_cog_attn = np.array([0.02009, 0.01989, 0.02007, 0.01989, 0.01995, 0.02014]) # was implemented wrong
# mla = np.array([0.01985, 0.0195, 0.01933, 0.01952, 0.01939])

mla_dint_no_bias = np.array([0.01967, 0.01955, 0.01958, 0.01943, 0.01929, 0.01924, 0.01934, 0.01934])
mla_dint_cog = np.array([0.01914, 0.01932, 0.01925, 0.01932, 0.01918]) # new baseline probably

mla_dint_cog_big = np.array([0.0189, 0.0187, 0.01883, 0.01895, 0.01879]) # better
# mla_dint_cog_noise_5 = np.array([0.01949, 0.01951, 0.01917, 0.01916, 0.01974, 0.01978, 0.01924, 0.01916]) # maybe worse
# mla_dint_cog_noise_10 = np.array([0.01935, 0.01936, 0.0194]) # worse
# mla_dint_cog_weighted_no_ortho = np.array([0.01937, 0.01907, 0.01935, 0.01912, 0.01936]) # same

# for features
mla_dint_cog_new_dataloader = np.array([0.01934, 0.0195, 0.01968, 0.01931, 0.01917, 0.01926, 0.01922, 0.01938]) # same-ish
basic_data = np.array([0.01894, 0.01877, 0.01886, 0.01884, 0.01886, 0.01868]) # standard znorm local
# diff znorm
diff_local_znorm_diff = np.array([0.01887, 0.01891, 0.01893, 0.01894, 0.01883])


basic_b = np.array([0.01979, 0.01989, 0.0199]) # only returns, standard znorm local
# diff znorm
basic_b_diff_local = np.array([0.01999, 0.0198, 0.0199]) # only returns, diff znorm local
basic_b_global = np.array([0.0191, 0.01888, 0.01895]) # only returns, global znorm much better
basic_b_global_diff = np.array([0.01893, 0.01902, 0.01908]) # only returns, global znorm diff worse than global
basic_b_global_group = np.array([0.01894, 0.01892, 0.0191]) # same as global
basic_b_global_g = np.array([0.01912, 0.01903, 0.01886]) # same as global
# variances
var_base_small = np.var(base_small, ddof=1)


# normal test (8 samples min)
_, p_value = stats.normaltest(base_small)
print(f"Normal test base_small p-value: {p_value}")

_, p_value = stats.normaltest(mla_dint_no_bias)
print(f"Normal test mla_dint_no_bias p-value: {p_value}")

# degrees of freedom
df_base_small = len(base_small) - 1

# # f-statistic to make sure same variance for t-test
# f_value = max(var_base_small, var_base_mha) / min(var_base_small, var_base_mha)
# p_value = 1.0-stats.f.cdf(f_value, df_base_small, df_base_mha)
# # F-statistic results
# print("F-test: base_small vs base_mha")
# print(f"F statistic: {f_value}")
# print(f"F statistic p-value: {p_value}")


# # t-test
# print("t-test: base_small vs base_mha")
# t_value, p_value = stats.ttest_ind(base_small, base_mha)
# print(f"T statistic: {t_value}")
# print(f"T statistic p-value: {p_value}")

# welchs t-test (not equal variance)
# print("Welch's t-test: base_small vs base_mha")
# t_value, p_value = stats.ttest_ind(base_small, base_mha, equal_var=False)
# print(f"Welch's T statistic: {t_value}")
# print(f"Welch's T statistic p-value: {p_value}")

# print("Welch's t-test: base_small vs base_mla_dint")
# t_value, p_value = stats.ttest_ind(base_small, base_mla_dint, equal_var=False)
# print(f"Welch's T statistic: {t_value}")
# print(f"Welch's T statistic p-value: {p_value}")

# print("Welch's t-test: base_small vs mla")
# t_value, p_value = stats.ttest_ind(base_small, mla, equal_var=False)
# print(f"Welch's T statistic: {t_value}")
# print(f"Welch's T statistic p-value: {p_value}")

# means
# print(f"Mean mla_dint_no_bias: {np.mean(mla_dint_no_bias)}")
# print(f"Mean mla_dint_cog: {np.mean(mla_dint_cog)}")

# print(f"Mean mla_dint_cog_big: {np.mean(mla_dint_cog_big)}")
# print(f"Mean mla_dint_cog_new_dataloader: {np.mean(mla_dint_cog_new_dataloader)}")



# print("Welch's t-test: mla_dint_no_bias vs mla_dint_cog")
# welch_t_test(mla_dint_no_bias, mla_dint_cog)

# print("Welch's t-test: mla_dint_cog vs mla_dint_cog_big")
# welch_t_test(mla_dint_cog, mla_dint_cog_big)

# print("Welch's t-test: mla_dint_cog vs mla_dint_cog_new_dataloader")
# welch_t_test(mla_dint_cog, mla_dint_cog_new_dataloader)

print(f"Mean basic_data: {np.mean(basic_data)}")
print(f"Mean diff_local_znorm_diff: {np.mean(diff_local_znorm_diff)}")
print("Welch's t-test: basic_data vs diff_local_znorm_diff")
welch_t_test(basic_data, diff_local_znorm_diff)

print(f"Mean basic_b: {np.mean(basic_b)}")
print(f"Mean basic_b_diff_local: {np.mean(basic_b_diff_local)}")
print("Welch's t-test: basic_b vs basic_b_diff_local")
welch_t_test(basic_b, basic_b_diff_local)

print(f"Mean basic_b_global: {np.mean(basic_b_global)}")
print("Welch's t-test: basic_b vs basic_b_global")
welch_t_test(basic_b, basic_b_global)

print(f"Mean basic_b_global_diff: {np.mean(basic_b_global_diff)}")
print("Welch's t-test: basic_b_global vs basic_b_global_diff")
welch_t_test(basic_b_global, basic_b_global_diff)

print(f"Mean basic_b_global_group: {np.mean(basic_b_global_group)}")
print("Welch's t-test: basic_b_global vs basic_b_global_group")
welch_t_test(basic_b_global, basic_b_global_group)

print(f"Mean basic_b_global_g: {np.mean(basic_b_global_g)}")
print("Welch's t-test: basic_b_global vs basic_b_global_g")
welch_t_test(basic_b_global, basic_b_global_g)