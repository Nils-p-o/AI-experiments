import numpy as np
import scipy.stats as stats

def welch_t_test(a, b):
    t_value, p_value = stats.ttest_ind(a, b, equal_var=False)
    print(f"Welch's T statistic: {t_value}")
    print(f"Welch's T statistic p-value: {p_value}")
    return

# TODO look into automatically getting results from log files
# TODO paired by seeds for paired tests
# TODO mann-whitney u and wilcoxon signed rank tests

global_norm_exclusive = False

if global_norm_exclusive:
    # by best achived val_loss_unseen

    base_3_global = np.array([0.5423, 0.5428, 0.544, 0.5438, 0.5437, 0.543, 0.5426, 0.5444])
    base_3_global_shorter = np.array([0.5445, 0.5432, 0.5439])
    # combined vol
    base_3_g_sh_vol = np.array([0.5428, 0.5435, 0.5434])
    base_3_g_sh_vol_c = np.array([0.5413, 0.5418, 0.5423]) # best so far
    # base_3_g_sh_vol_chlov = np.array([0.5428, 0.5422, 0.5428])

    # sma and ema
    base_3_g_sh_vol_c_sma_c = np.array([0.5415, 0.5415, 0.5417]) # same-ish
    base_3_g_sh_vol_c_sma_chlov = np.array([0.5362, 0.5365, 0.5353]) # better
    base_3_g_sh_vol_c_ema_chlov = np.array([0.5236, 0.5243, 0.5235, 0.5239, 0.5247, 0.5241, 0.5247, 0.5241]) # best so far
    # sometimes called 3_base_ema (current baseline to be compared with)

    # ema_chlov_mfi = np.array([0.5241, 0.524, 0.5245])
    # ema_chlov_mfi_ch = np.array([0.5242, 0.5239, 0.5238])

    ema_chlov_vpt_chlo = np.array([0.5228, 0.5229, 0.5222])
    # ema_chlov_vpt_ch = np.array([0.5235, 0.5234, 0.5248])
    # ema_chlov_vpt_m = np.array([0.5242, 0.525, 0.5237])
    # ema_chlov_vpt_m_ch = np.array([0.5242, 0.5239, 0.5234])
    # ema_chlov_atr_div = np.array([0.523, 0.523, 0.5232, 0.525, 0.5239])

    # ema_chlov_ppo_c = np.array([0.523, 0.5237, 0.5224])
    # ema_chlov_ppo_chlo = np.array([0.5217, 0.5236, 0.522, 0.5218, 0.5238])
    ema_chlov_ppo_chlov = np.array([0.5206, 0.5202, 0.5197, 0.5199])
    # ema_chlov_rsi_c = np.array([0.5247, 0.5228, 0.5233, 0.5238, 0.5231])
    # ema_chlov_rsi_chlo = np.array([0.524, 0.5238, 0.5255])

    ema_chlov_clv = np.array([0.5229, 0.523, 0.5229, 0.5242, 0.5234])
    # ema_chlov_vix_c_simple = np.array([0.5237, 0.5237, 0.5233])
    ema_chlov_vix_chlo_simple = np.array([0.5226, 0.523, 0.5229])
    # ema_chlov_vix_chlo_com = np.array([0.5239, 0.5225, 0.5228])

    ema_USt = np.array([0.524, 0.5236, 0.5231])
    # ema_gold_c_ch = np.array([0.524, 0.5255, 0.5236])
    ema_gold_chlov = np.array([0.5231, 0.5237, 0.5226])
    # ema_crude_oil_c = np.array([0.5249, 0.5238, 0.5242])
    ema_copper_chlov = np.array([0.5237, 0.5233, 0.5233])
    ema_crude_oil_chlov = np.array([0.5238, 0.5237, 0.5239])
    # ema_silver_chlov = np.array([0.5236, 0.524, 0.524])
    # ema_usd_index_chlov = np.array([0.5237, 0.5247, 0.523])
    # ema_alphas = np.array([0.523, 0.5241, 0.524])
    # ema_rel_vol_div = np.array([0.5233, 0.524, 0.5242])
    # ema_rel_vol_sub = np.array([0.5246, 0.5236, 0.5251])
    ema_old_adr = np.array([0.5227, 0.5237, 0.5227])
    # ema_bb_ret_c = np.array([0.524, 0.5244, 0.5234])
    # ema_bb_price_c_signal = np.array([0.525, 0.5229, 0.5237, 0.5235])
    # ema_macd_c = np.array([0.528, 0.5302, 0.5248])
    # ema_chaikin_old = np.array([0.5243, 0.5236, 0.5241, 0.5248])
    ema_ad = np.array([0.5238, 0.5234, 0.5223])
    # ema_ad_old = np.array([0.5251, 0.5245, 0.5236])
    # ema_chaikin_standard = np.array([0.5241, 0.5248, 0.5247])
    # ema_stochastic_oscillator = np.array([0.5249, 0.5244, 0.5236])
    # ema_vpt_old_c = np.array([0.5235, 0.5247, 0.5237])


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


    print(f"Mean base_3_g_sh_vol_c_ema_chlov: {np.mean(base_3_g_sh_vol_c_ema_chlov)}")
    # print("Welch's t-test: base_3_g_sh_vol_c vs base_3_g_sh_vol_c_ema_chlov")
    # welch_t_test(base_3_g_sh_vol_c, base_3_g_sh_vol_c_ema_chlov) # better

    # print("Welch's t-test: base_3_g_sh_vol_c_sma_chlov vs base_3_g_sh_vol_c_ema_chlov")
    # welch_t_test(base_3_g_sh_vol_c_sma_chlov, base_3_g_sh_vol_c_ema_chlov) # best so far


    print(f"Mean ema_chlov_vpt_chlo: {np.mean(ema_chlov_vpt_chlo)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_vpt")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_vpt_chlo) # 0.004

    # print(f"Mean ema_chlov_atr_div: {np.mean(ema_chlov_atr_div)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_atr_div")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_atr_div) # same-ish (0.28)

    # print(f"Mean ema_chlov_ppo_c: {np.mean(ema_chlov_ppo_c)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_ppo_c")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_ppo_c) # 0.175 better?

    # print(f"Mean ema_chlov_ppo_chlo: {np.mean(ema_chlov_ppo_chlo)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_ppo_chlo")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_ppo_chlo) # 0.06 better

    print(f"Mean ema_chlov_ppo_chlov: {np.mean(ema_chlov_ppo_chlov)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_ppo_chlov")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_ppo_chlov) # 1e-6 better def

    # print(f"Mean ema_chlov_rsi_c: {np.mean(ema_chlov_rsi_c)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_rsi_c")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_rsi_c) # 0.524

    # print(f"Mean ema_chlov_rsi_chlo: {np.mean(ema_chlov_rsi_chlo)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_rsi_chlo")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_rsi_chlo) # 0.62

    print(f"Mean ema_chlov_clv: {np.mean(ema_chlov_clv)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_clv")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_clv) # 0.025 better

    # print(f"Mean ema_chlov_vix_c_simple: {np.mean(ema_chlov_vix_c_simple)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_vix_c_simple")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_vix_c_simple) # 0.03

    print(f"Mean ema_chlov_vix_chlo_simple: {np.mean(ema_chlov_vix_chlo_simple)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_vix_chlo_simple")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_vix_chlo_simple) # 0.00019 better

    # print(f"Mean ema_chlov_vix_chlo_com: {np.mean(ema_chlov_vix_chlo_com)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chlov_vix_chlo_com")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chlov_vix_chlo_com) # 0.11

    print(f"Mean ema_USt: {np.mean(ema_USt)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_USt")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_USt) # 0.15 better?

    # print(f"Mean ema_gold_c_ch: {np.mean(ema_gold_c_ch)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_gold_c_ch")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_gold_c_ch) # 0.52 

    print(f"Mean ema_gold_chlov: {np.mean(ema_gold_chlov)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_gold_chlov")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_gold_chlov) # 0.069 better?

    # print(f"Mean ema_crude_oil_c: {np.mean(ema_crude_oil_c)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_crude_oil_c")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_crude_oil_c) # 0.64

    print(f"Mean ema_crude_oil_chlov: {np.mean(ema_crude_oil_chlov)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_crude_oil_chlov")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_crude_oil_chlov) # 0.099 better?

    print(f"Mean ema_copper_chlov: {np.mean(ema_copper_chlov)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_copper_chlov")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_copper_chlov) # 0.0125 better

    # print(f"Mean ema_silver_chlov: {np.mean(ema_silver_chlov)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_silver_chlov")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_silver_chlov) # 0.272 better?

    # print(f"Mean ema_usd_index_chlov: {np.mean(ema_usd_index_chlov)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_usd_index_chlov")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_usd_index_chlov) # 0.59

    # print(f"Mean ema_alphas: {np.mean(ema_alphas)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_alphas")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_alphas) # 0.36

    # print(f"Mean ema_rel_vol_div: {np.mean(ema_rel_vol_div)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_rel_vol_div")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_rel_vol_div) # 0.43

    # print(f"Mean ema_rel_vol_sub: {np.mean(ema_rel_vol_sub)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_rel_vol_sub")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_rel_vol_sub) # 0.55

    print(f"Mean ema_old_adr: {np.mean(ema_old_adr)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_old_adr")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_old_adr) # 0.06 better?

    # print(f"Mean ema_bb_ret: {np.mean(ema_bb_ret)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_bb_ret")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_bb_ret) # 0.62

    # print(f"Mean ema_bb_price_c_signal: {np.mean(ema_bb_price_c_signal)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_bb_price_c_signal")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_bb_price_c_signal) # 0.51

    # print(f"Mean ema_macd_c: {np.mean(ema_macd_c)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_macd_c")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_macd_c) # 0.15 worse

    # print(f"Mean ema_chaikin_old: {np.mean(ema_chaikin_old)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chaikin_old")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chaikin_old) # 0.77 worse ish

    print(f"Mean ema_ad: {np.mean(ema_ad)}")
    print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_ad")
    welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_ad) # 0.15 better

    # print(f"Mean ema_ad_old: {np.mean(ema_ad_old)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_ad_old")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_ad_old) # 0.58 worse

    # print(f"Mean ema_chaikin_standard: {np.mean(ema_chaikin_standard)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_chaikin_standard")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_chaikin_standard) # 0.19 worse

    # print(f"Mean ema_stochastic_oscillator: {np.mean(ema_stochastic_oscillator)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_stochastic_oscillator")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_stochastic_oscillator) # 0.68 worse

    # print(f"Mean ema_vpt_old_c: {np.mean(ema_vpt_old_c)}")
    # print("Welch's t-test: base_3_g_sh_vol_c_ema_chlov vs ema_vpt_old_c")
    # welch_t_test(base_3_g_sh_vol_c_ema_chlov, ema_vpt_old_c) # 0.74


# local norm feats
# clv, vpt are questionable

# price emas, vol, ppo, ret emas  -- are solid

bee6_local_price_emas = np.array([0.3707, 0.3706, 0.3689])
bee6_local_price_emas_vol = np.array([0.3588, 0.3581, 0.3583])
bee6_local_price_emas_vol_ppo = np.array([0.3506, 0.3531, 0.3562])
bee6_local_price_emas_vol_ppo_clv = np.array([0.3531, 0.3483, 0.3539])
bee6_local_price_emas_vol_ppo_clv_vpt = np.array([0.3444, 0.3512, 0.3492])
bee6_local_price_emas_vol_ppo_clv_ret_emas = np.array([0.3357, 0.3295, 0.3335])
bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas = np.array([0.3386, 0.3353]) # worse?
bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed = np.array([0.333, 0.3267, 0.3332]) # best so far

# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_ret = np.array([0.3276, 0.3304, 0.3193])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_price = np.array([0.3288, 0.3274, 0.3239])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_combined = np.array([0.3278, 0.3219, 0.3286])
bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_combined_hist = np.array([0.3211, 0.3226, 0.3246, 0.3256]) # best of bb

bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_atr_div = np.array([0.3277, 0.3292, 0.3262])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_chaikin = np.array([0.3381, 0.3342, 0.3339])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_chaikin_old = np.array([0.3344, 0.3361, 0.3326])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_ad = np.array([0.328, 0.3294, 0.339])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_adr = np.array([0.3338, 0.3275, 0.3301])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_macd_c = np.array([0.332, 0.3332])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_macd_c_hist = np.array([0.335, 0.3348])


# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_mfi = np.array([0.3345, 0.3343, 0.3351])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_mfi_ch = np.array([0.3343, 0.3283, 0.332])

bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_ret_vol_chlov = np.array([0.319, 0.3229, 0.3259])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_rsi = np.array([0.3355, 0.3401])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_stoch_osc = np.array([0.3303, 0.3373])
# bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_vpt_old = np.array([0.3323, 0.3346, 0.3288])


print(f"Mean bee6_local_price_emas: {np.mean(bee6_local_price_emas)}")
print(f"Mean bee6_local_price_emas_vol: {np.mean(bee6_local_price_emas_vol)}")
print("Welch's t-test: bee6_local_price_emas vs bee6_local_price_emas_vol")
welch_t_test(bee6_local_price_emas, bee6_local_price_emas_vol) # 0.0009 better

# print(f"Mean bee6_local_price_emas_vol_ppo: {np.mean(bee6_local_price_emas_vol_ppo)}")
# print("Welch's t-test: bee6_local_price_emas_vol vs bee6_local_price_emas_vol_ppo")
# welch_t_test(bee6_local_price_emas_vol, bee6_local_price_emas_vol_ppo) # 0.085 better

# print(f"Mean bee6_local_price_emas_vol_ppo_clv: {np.mean(bee6_local_price_emas_vol_ppo_clv)}")
# print("Welch's t-test: bee6_local_price_emas_vol vs bee6_local_price_emas_vol_ppo_clv")
# welch_t_test(bee6_local_price_emas_vol, bee6_local_price_emas_vol_ppo_clv) # 0.061 better

# print(f"Mean bee6_local_price_emas_vol_ppo_clv_vpt: {np.mean(bee6_local_price_emas_vol_ppo_clv_vpt)}")
# print("Welch's t-test: bee6_local_price_emas_vol vs bee6_local_price_emas_vol_ppo_clv_vpt")
# welch_t_test(bee6_local_price_emas_vol, bee6_local_price_emas_vol_ppo_clv_vpt) # 0.036 better

print(f"Mean bee6_local_price_emas_vol_ppo_clv_ret_emas: {np.mean(bee6_local_price_emas_vol_ppo_clv_ret_emas)}")
print("Welch's t-test: bee6_local_price_emas_vol vs bee6_local_price_emas_vol_ppo_clv_ret_emas")
welch_t_test(bee6_local_price_emas_vol, bee6_local_price_emas_vol_ppo_clv_ret_emas) # 0.0045


print(f"Mean bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_combined_hist: {np.mean(bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_combined_hist)}")
print("Welch's t-test: bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed vs bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_combined_hist")
welch_t_test(bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed, bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_bb_c_combined_hist) # 0.05 best of the block


print(f"Mean bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_atr_div: {np.mean(bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_atr_div)}")
print("Welch's t-test: bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed vs bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_atr_div")
welch_t_test(bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed, bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_atr_div) # 0.26 maybe

print(f"Mean bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_ret_vol_chlov: {np.mean(bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_ret_vol_chlov)}")
print("Welch's t-test: bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed vs bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_ret_vol_chlov")
welch_t_test(bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_fixed, bee6_local_price_emas_vol_ppo_clv_vpt_ret_emas_ret_vol_chlov) # 0.046 better
