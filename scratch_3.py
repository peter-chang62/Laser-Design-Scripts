# # initial polarization state
# x = np.array([1, 1]) / np.sqrt(2)

# # x = np.array([1, 0])
# # x = rot(np.pi / 4) @ M_qwp @ inv(rot(np.pi / 4)) @ x

# Theta_hwp = np.linspace(0, np.pi / 2, 125)
# Theta_qwp = np.linspace(0, 2 * np.pi, 100)
# Output = np.zeros((Theta_qwp.size, Theta_hwp.size, 2), dtype=complex)

# for i, theta_qwp in enumerate(Theta_qwp):
#     for j, theta_hwp in enumerate(Theta_hwp):
#         y = freespace(x, theta_qwp, theta_hwp)
#         Output[i, j] = y

# fig, ax = plt.subplots(1, 1)
# img = ax.pcolormesh(
#     Theta_hwp * 180 / np.pi, Theta_qwp * 180 / np.pi, np.sum(abs(Output) ** 2, axis=2)
# )
# ax.set_title("free space transmission")
# ax.set_xlabel("$\\mathrm{\\lambda /2}$ deg")
# ax.set_ylabel("$\\mathrm{\\lambda /4}$ deg")
# plt.colorbar(img, ax=ax)
