import matplotlib.pyplot as plt

def debug_plot_points(image, points, title, filename, points2 = None):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='inferno', origin='upper')
    if points is not None and points.shape[0] > 0:
        # Assumes points are (N, 2) with (y, x) columns
        ax.scatter(points[:, 1], points[:, 0], c='cyan', s=15, zorder=2, alpha = 0.50)
    if points2 is not None and points.shape[0] > 0:
        ax.scatter(points2[:, 1], points2[:, 0], c='magenta', s=15, zorder=2, alpha = 0.50)
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    plt.savefig(f"output/{filename}")
    plt.close(fig)


def debug_plot_path(image, points, title, filename):
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='inferno', origin='upper')
    if points is not None and points.shape[0] > 1:
        # Assumes points are (N, 2) with (y, x) columns
        ax.plot(points[:, 1], points[:, 0], 'r-', zorder=2, linewidth=1) # Draw lines
        ax.scatter(points[:, 1], points[:, 0], c='cyan', s=15, zorder=3) # Draw points
    ax.set_title(title, fontsize=8)
    ax.set_aspect('equal')
    plt.savefig(f"output/{filename}")
    plt.close(fig)