# # %% -------------- load absorption coefficients ------------------------------
# sigma = np.genfromtxt("Ansys/er_cross_section_fig_6_1.txt")
# a = sigma[3:][:, :2]
# e = sigma[3:][:, [0, 2]]

# sigma_pump = sigma[0, 1]

# spl_a = InterpolatedUnivariateSpline(c / a[:, 0][::-1], a[:, 1][::-1], ext="zeros")
# spl_e = InterpolatedUnivariateSpline(c / e[:, 0][::-1], e[:, 1][::-1], ext="zeros")

# # %% -------------- load dispersion coefficients ------------------------------
# frame = pd.read_excel("nLIGHT_Er110-4_125-PM_simulated_GVD_dispersion.xlsx")
# gvd = frame.to_numpy()[:, :2][1:].astype(float)

# wl = gvd[:, 0] * 1e-9
# omega = 2 * np.pi * c / wl
# omega0 = 2 * np.pi * c / 1550e-9
# polyfit = np.polyfit(omega - omega0, gvd[:, 1], deg=3)
# polyfit = polyfit[::-1]  # lowest order first

# # %% ------------- pulse ------------------------------------------------------
# n = 256
# v_min = c / 2000e-9
# v_max = c / 1000e-9
# v0 = c / 1550e-9
# e_p = 10e-12
# t_fwhm_short = 250e-15
# t_fwhm_long = 2e-12
# min_time_window = 20e-12
# pulse_short = pynlo.light.Pulse.Sech(
#     n,
#     v_min,
#     v_max,
#     v0,
#     e_p,
#     t_fwhm_short,
#     min_time_window,
#     alias=2,
# )

# pulse_long = pynlo.light.Pulse.Sech(
#     n,
#     v_min,
#     v_max,
#     v0,
#     e_p,
#     t_fwhm_long,
#     min_time_window,
#     alias=2,
# )

# # %% ------------- fiber ------------------------------------------------------
# fiber = pynlo.materials.SilicaFiber()
# fiber.set_beta_from_beta_n(v0, polyfit)
# fiber.gamma = 4 / (W * km)

# # %% ------------- edfa -------------------------------------------------------
# amp = amplify(
#     pulse_short,
#     pulse_short,
#     fiber,
#     0,
#     50e-3,
#     5,
#     sigma_pump,
#     spl_a(pulse_short.v_grid),
#     spl_e(pulse_short.v_grid),
#     error=1e-3,
# )

# # %% --------- look at results! -----------------------------------------------
# amp.sim_fwd.plot("wvl", num="forward")
# amp.sim_bck.plot("wvl", num="backward")

# fig, ax = plt.subplots(1, 1)
# ax.plot(amp.sim_fwd.z, amp.Pp)
# ax_2 = ax.twinx()
# ax_2.plot(amp.sim_fwd.z, amp.n2_n, "C1")
# ax_2.set_ylim(ymax=1)
