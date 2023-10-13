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

# # %% ------------- pulse ------------------------------------------------------¬
# n = 256
# v_min = c / 2000e-9
# v_max = c / 1000e-9
# v0 = c / 1550e-9
# e_p = 10e-12
# t_fwhm = 250e-15
# min_time_window = 20e-12
# pulse = pynlo.light.Pulse.Sech(
#     n,
#     v_min,
#     v_max,
#     v0,
#     e_p,
#     t_fwhm,
#     min_time_window,
#     alias=2,
# )

# # %% ------------- fiber ------------------------------------------------------
# fiber = pynlo.materials.SilicaFiber()
# fiber.set_beta_from_beta_n(v0, polyfit)
# fiber.gamma = 4 / (W * km)

# # %% -------- edfa pump power parameter sweep  --------------------------------
# length = 5
# start = 1e-3
# stop = 50e-3
# step = 1e-3 / 2.0
# Pp = np.arange(start, stop + step, step)
# AMP = []
# for n, pp in enumerate(tqdm(Pp)):
#     amp = amplify(
#         pulse,
#         None,
#         fiber,
#         pp,
#         pp,
#         length,
#         sigma_pump,
#         spl_a(pulse.v_grid),
#         spl_e(pulse.v_grid),
#         error=1e-3,
#     )
#     AMP.append(amp)

# # %% ------------------------- look at results! -------------------------------
# g_dB = np.asarray([i.g_dB_fwd for i in AMP])

# fig, ax = plt.subplots(1, 1)
# ax.plot(Pp * 1e3, g_dB)
# ax.set_xlabel("pump power (mW)")
# ax.set_ylabel("signal gain (dB)")
# fig.tight_layout()

# fig, ax = plt.subplots(3, 1, figsize=np.array([4.67, 8.52]))
# ax[:] = ax[::-1]
# idx_min = g_dB.argmin()
# idx_half = abs(g_dB - g_dB.max() / 2).argmin()
# idx_max = g_dB.argmax()

# ax[0].plot(AMP[idx_min].sim_fwd.z, AMP[idx_min].n2_n)
# ax[0].set_ylabel("$\\mathrm{n_2/n_1}$")
# ax_2 = ax[0].twinx()
# ax_2.plot(AMP[idx_min].sim_fwd.z, AMP[idx_min].Pp * 1e3, "C1")
# ax_2.set_ylabel("pump power (mW)")
# ax[0].set_xlabel("position (m)")
# ax[0].set_ylim(ymax=1)

# ax[1].plot(AMP[idx_half].sim_fwd.z, AMP[idx_half].n2_n)
# ax[1].set_ylabel("$\\mathrm{n_2/n_1}$")
# ax_2 = ax[1].twinx()
# ax_2.plot(AMP[idx_half].sim_fwd.z, AMP[idx_half].Pp * 1e3, "C1")
# ax_2.set_ylabel("pump power (mW)")
# ax[1].set_xlabel("position (m)")
# ax[1].set_ylim(ymax=1)

# ax[2].plot(AMP[idx_max].sim_fwd.z, AMP[idx_max].n2_n)
# ax[2].set_ylabel("$\\mathrm{n_2/n_1}$")
# ax_2 = ax[2].twinx()
# ax_2.plot(AMP[idx_max].sim_fwd.z, AMP[idx_max].Pp * 1e3, "C1")
# ax_2.set_ylabel("pump power (mW)")
# ax[2].set_xlabel("position (m)")
# ax[2].set_ylim(ymax=1)

# fig.tight_layout()

# AMP[idx_min].sim_fwd.plot("wvl", num="minimum")
# AMP[idx_half].sim_fwd.plot("wvl", num="half")
# AMP[idx_max].sim_fwd.plot("wvl", num="max")
