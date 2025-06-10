import numpy as np
import scipy.stats as stats

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
base_3_global = np.array([0.5423, 0.5428, 0.544, 0.5438, 0.5437, 0.543, 0.5426, 0.5444])
base_3_global_shorter = np.array([0.5445, 0.5432, 0.5439])


# normal test (8 samples min)
_, p_value = stats.normaltest(base_3_global)
print(f"Normal test base_small p-value: {p_value}")


# degrees of freedom

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
# print("Welch's t-test: base_small vs base_mla_dint")
# t_value, p_value = stats.ttest_ind(base_small, base_mla_dint, equal_var=False)
# print(f"Welch's T statistic: {t_value}")
# print(f"Welch's T statistic p-value: {p_value}")

# means

print(f"Mean base_3_global: {np.mean(base_3_global)}")
print(f"Mean base_3_global_shorter: {np.mean(base_3_global_shorter)}")
print("Welch's t-test: base_3_global vs base_3_global_shorter")
welch_t_test(base_3_global, base_3_global_shorter)