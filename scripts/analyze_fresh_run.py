import numpy as np
from scipy import stats

tasks = [
  ('03560426', 3, 0.0, 69.0, 71.0, 40.0),
  ('0becf7df', 3, 0.0, 79.0, 77.0, 80.0),
  ('12eac192', 4, 0.0, 76.6, 70.3, 76.6),
  ('17cae0c1', 4, 0.0, 44.4,  0.0, 11.1),
  ('2685904e', 6, 0.0, 86.0, 88.0, 84.0),
  ('0ca9ddb6', 3, 0.0, 70.4, 85.2, 55.6),
  ('29623171', 3, 0.0, 62.8, 81.8, 85.1),
  ('ed74f2f2', 6, 0.0,  0.0, 22.2, 22.2),
  ('77fdfe62', 3, 0.0, 44.4, 27.8, 30.6),
  ('1cf80156', 3, 0.0,  8.3, 54.2,  0.0),
  ('67385a82', 4, 0.0, 92.0, 64.0, 96.0),
  ('d017b73f', 4, 0.0, 58.3, 50.0, 58.3),
  ('6855a6e4', 3, 0.0, 93.3, 88.4, 93.8),
  ('b8cdaf2b', 4, 0.0, 86.4, 92.6, 82.7),
  ('73c3b0d8', 4, 0.0, 93.8, 91.7, 26.0),
  ('b7cb93ac', 3, 0.0,  0.0,  0.0,  0.0),
  ('db3e9e38', 2, 0.0, 77.8, 56.8, 72.8),
  ('3af2c5a8', 3, 0.0, 16.7, 25.0, 16.7),
  ('e57337a4', 3, 0.0, 11.1, 77.8,  0.0),
  ('45737921', 3, 0.0, 79.2, 75.0, 81.9),
  ('7c8af763', 3, 0.0, 72.0, 52.0, 74.0),
  ('e0fb7511', 3, 0.0, 92.9, 80.5, 88.8),
  ('c48954c1', 3, 0.0,  0.0, 14.8, 29.6),
  ('af24b4cc', 3, 0.0, 80.0, 50.0, 80.0),
  ('05f2a901', 3, 0.0, 74.5, 89.1, 90.0),
]

llama = [t[2] for t in tasks]
rh = [t[3] for t in tasks]
s1 = [t[4] for t in tasks]
s2 = [t[5] for t in tasks]
oracle_s1_s2 = [max(t[4], t[5]) for t in tasks]
oracle_all = [max(t[3], t[4], t[5]) for t in tasks]

print(f"Mean Raw Llama:     {np.mean(llama):.2f}% ± {stats.sem(llama):.2f}%")
print(f"Mean Pure RH:       {np.mean(rh):.2f}% ± {stats.sem(rh):.2f}%")
print(f"Mean S1 Reflex:     {np.mean(s1):.2f}% ± {stats.sem(s1):.2f}%")
print(f"Mean S2 TTA:        {np.mean(s2):.2f}% ± {stats.sem(s2):.2f}%")
print(f"Oracle(S1, S2):     {np.mean(oracle_s1_s2):.2f}% ± {stats.sem(oracle_s1_s2):.2f}%")
print(f"Oracle(RH, S1, S2): {np.mean(oracle_all):.2f}% ± {stats.sem(oracle_all):.2f}%")

# S1 vs RH
t_s1_rh, p_s1_rh = stats.ttest_rel(s1, rh)
diff_s1_rh = np.array(s1) - np.array(rh)
non_zero = diff_s1_rh[diff_s1_rh != 0]
w_s1_rh, pw_s1_rh = stats.wilcoxon(non_zero)
print(f"S1 vs RH: t={t_s1_rh:.4f}, p={p_s1_rh:.4f}, Wilcoxon W={w_s1_rh:.1f}, p={pw_s1_rh:.4f}")

# S2 vs RH
t_s2_rh, p_s2_rh = stats.ttest_rel(s2, rh)
diff_s2_rh = np.array(s2) - np.array(rh)
non_zero_2 = diff_s2_rh[diff_s2_rh != 0]
w_s2_rh, pw_s2_rh = stats.wilcoxon(non_zero_2)
print(f"S2 vs RH: t={t_s2_rh:.4f}, p={p_s2_rh:.4f}, Wilcoxon W={w_s2_rh:.1f}, p={pw_s2_rh:.4f}")

# Oracle vs RH
t_orc_rh, p_orc_rh = stats.ttest_rel(oracle_s1_s2, rh)
diff_orc_rh = np.array(oracle_s1_s2) - np.array(rh)
non_zero_orc = diff_orc_rh[diff_orc_rh != 0]
w_orc_rh, pw_orc_rh = stats.wilcoxon(non_zero_orc)
print(f"Oracle(S1,S2) vs RH: t={t_orc_rh:.4f}, p={p_orc_rh:.4f}, Wilcoxon W={w_orc_rh:.1f}, p={pw_orc_rh:.4f}, delta=+{np.mean(oracle_s1_s2)-np.mean(rh):.2f}%")
orc_wins = sum(1 for o, r in zip(oracle_s1_s2, rh) if o > r)
rh_orc_wins = sum(1 for o, r in zip(oracle_s1_s2, rh) if r > o)
orc_ties = sum(1 for o, r in zip(oracle_s1_s2, rh) if o == r)
print(f"Oracle vs RH Record: {orc_wins} Wins / {orc_ties} Ties / {rh_orc_wins} Losses")

s1_wins = sum(1 for s, r in zip(s1, rh) if s > r)
rh_wins = sum(1 for s, r in zip(s1, rh) if r > s)
ties = sum(1 for s, r in zip(s1, rh) if s == r)
print(f"S1 vs RH Record: {s1_wins} Wins / {ties} Ties / {rh_wins} Losses")

s2_wins = sum(1 for s, r in zip(s2, rh) if s > r)
rh_wins_s2 = sum(1 for s, r in zip(s2, rh) if r > s)
ties_s2 = sum(1 for s, r in zip(s2, rh) if s == r)
print(f"S2 vs RH Record: {s2_wins} Wins / {ties_s2} Ties / {rh_wins_s2} Losses")

s1_s2_wins = sum(1 for s, s_2 in zip(s1, s2) if s > s_2)
s2_s1_wins = sum(1 for s, s_2 in zip(s1, s2) if s_2 > s)
s1_s2_ties = sum(1 for s, s_2 in zip(s1, s2) if s == s_2)
print(f"S1 vs S2 Record: {s1_s2_wins} S1 Wins / {s1_s2_ties} Ties / {s2_s1_wins} S2 Wins")

t_casc_s1, p_casc_s1 = stats.ttest_rel(oracle_s1_s2, s1)
diff_casc_s1 = np.array(oracle_s1_s2) - np.array(s1)
non_zero_s1 = diff_casc_s1[diff_casc_s1 != 0]
w_casc_s1, pw_casc_s1 = stats.wilcoxon(non_zero_s1)
print(f"Cascaded vs S1: t={t_casc_s1:.4f}, p={p_casc_s1:.4f}, Wilcoxon W={w_casc_s1:.1f}, p={pw_casc_s1:.4f}, delta=+{np.mean(oracle_s1_s2)-np.mean(s1):.2f}%")

