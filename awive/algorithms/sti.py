"""Space Time Image Velocimetry."""

import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.config import Config
from awive.config import Stiv as StivConfig
from awive.loader import Loader, make_loader
from awive.preprocess.correct_image import Formatter

LOG = logging.getLogger(__name__)


class STIV:
    """Space Time Image Velocimetry."""

    def __init__(
        self,
        config: StivConfig,
        loader: Loader,
        formatter: Formatter,
        lines: list[int],
        images_dp: Path | None = None,
    ) -> None:
        """Initialize STIV."""
        self.cnt = 0
        # Shall be initialized later
        self.lines = [int(line * formatter.resolution) for line in lines]
        self.lines_range = [
            (
                int(lrange[0] * formatter.resolution),
                int(lrange[1] * formatter.resolution),
            )
            for lrange in config.lines_range
        ]
        self.images_dp = images_dp
        self.config = config
        self.loader = loader
        self.formatter = formatter
        self.stis_qnt = len(lines)
        LOG.debug(f"Number of frames: {self.loader.total_frames}")
        LOG.debug(f"Frames per second: {self.loader.fps}")
        LOG.debug(f"Pixels per meter: {self.formatter.ppm}")
        t0 = time.process_time()
        self.stis: list[NDArray] = self.generate_st_images()
        t1 = time.process_time()

        # create filter window
        w_size = self.config.filter_window
        w_mn = (1 - np.cos(2 * math.pi * np.arange(w_size) / w_size)) / 2
        w_mn = np.tile(w_mn, (w_size, 1))
        self._filter_win = w_mn * w_mn.T

        # vertical and horizontal filter width
        self._vh_filter = 1
        self._polar_filter_width = self.config.polar_filter_width
        LOG.debug(f"- generate_st_images: {t1 - t0}")

    def save(self, filename: str, image: NDArray) -> None:
        """Save image to disk."""
        if self.images_dp is not None:
            np.save(str(self.images_dp / filename), image)

    def get_velocity(self, angle: float) -> float:
        """Given STI pattern angle, calculate velocity.

        Args:
            angle: angle in radians

        Returns:
            velocity in m/s
        """
        return (
            math.tan(angle)
            * self.loader.fps
            / self.formatter.ppm
            / self.formatter.resolution
        )

    def generate_st_images(self) -> list[NDArray]:
        """Generate space time images."""
        # initialize set of sti images
        stis = [[] for _ in range(self.stis_qnt)]

        # generate all lines
        while self.loader.has_images():
            image = self.loader.read()
            if image is None:
                break
            image = self.formatter.apply_distortion_correction(image)
            image = self.formatter.apply_roi_extraction(image)
            image = self.formatter.apply_resolution(image)

            assert len(self.lines) == len(self.lines_range)
            for i, (line_pos, line_range) in enumerate(
                zip(self.lines, self.lines_range)
            ):
                row = image[line_pos, line_range[0] : line_range[1]]
                stis[i].append(row)

        for i in range(self.stis_qnt):
            stis[i] = np.array(stis[i])  # pyright: ignore[reportCallIssue, reportArgumentType]
            self.save(f"sti_{i:04}.npy", stis[i])  # pyright: ignore[reportArgumentType]
        return stis  # pyright: ignore[reportReturnType]

    @staticmethod
    def _get_main_freqs(isd: NDArray) -> list[int]:
        main_freqs = []
        main_freqs.append(np.argmax(isd))
        if main_freqs[0] < len(isd) / 2:
            main_freqs.append(main_freqs[0] + len(isd) / 2)
        else:
            main_freqs.append(main_freqs[0] - len(isd) / 2)
        return main_freqs

    def _apply_angle(
        self,
        mask: NDArray,
        freq: float,
        isd: NDArray,
    ) -> NDArray:
        x = int(freq - self._polar_filter_width)
        y = int(freq + self._polar_filter_width)
        if x < 0:
            x = 0
            mask[x + len(isd) :, :] = 1
        elif y > len(isd):
            y = len(isd)
            mask[: y - len(isd), :] = 1
        mask[x:y, :] = 1
        return mask

    def _generate_polar_mask(self, polar_img: NDArray) -> NDArray:
        # calculate Integral Spectrum Distribution
        isd = np.sum(polar_img.T, axis=0)
        main_freqs = self._get_main_freqs(isd)
        mask = np.zeros(polar_img.shape)

        mask = self._apply_angle(mask, main_freqs[0], isd)
        return self._apply_angle(mask, main_freqs[1], isd)

    def _process_sti(self, image: np.ndarray) -> tuple[float, float]:
        """Process sti image."""
        # image = cv2.medianBlur(image, 5)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self.config.ksize)
        sobelt = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self.config.ksize)
        if sobelx.sum() == 0 and sobelt.sum() == 0:
            LOG.warning("gradients are zero")
            return 0, 0

        j_xx = (sobelx * sobelx).sum()
        j_tt = (sobelt * sobelt).sum()
        j_xt = (sobelx * sobelt).sum()
        angle = math.atan2(2 * j_xt, j_tt - j_xx) / 2
        coherence = math.sqrt((j_tt - j_xx) ** 2 + 4 * j_xt**2) / (j_xx + j_tt)
        return angle, coherence

    @staticmethod
    def _get_new_point(
        point: tuple[int, int],
        angle: float,
        length: float,
    ) -> tuple[int, int]:
        """Will plot the line on a 10 x 10 plot.

        Args:
            point: Point (x, y) where you want to start the line.
            angle: Angle you want your end point at in degrees.
            length: Length of the line you want to plot.

        Returns:
            Point - Tuple (x, y).
        """
        # unpack the first point
        x, y = point
        # find the end point
        endy = length * math.cos(angle)
        endx = length * math.sin(angle)
        return int(endx + x), int(-endy + y)

    def _draw_angle(
        self,
        image: NDArray,
        angle: float,
        position: tuple[int, int],
        thick: int = 1,
        amplitud: int = 10,
    ) -> NDArray:
        new_point = self._get_new_point(position, angle, amplitud)
        cv2.line(image, new_point, position, 255, thick)
        return image

    @staticmethod
    def _conv2d(a: NDArray, f: NDArray) -> NDArray:
        """Convolve 2d array with a filter.

        https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy

        Args:
            a: 2d array
            f: filter

        Returns:
            Convolved array
        """
        s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
        strd = np.lib.stride_tricks.as_strided
        sub_m = strd(a, shape=s, strides=a.strides * 2)
        return np.einsum("ij,ijkl->kl", f, sub_m)

    @staticmethod
    def _to_polar_system(img: np.ndarray, option: str = "convert") -> NDArray:
        """Transform 2d image to polar system.

        Args:
            img: 2d image
            option: "convert" or "invert"

        Returns:
            Polar image
        """
        if option == "invert":
            flag = cv2.WARP_INVERSE_MAP
        else:
            flag = cv2.WARP_FILL_OUTLIERS

        # TODO: I should add padding

        row, col = img.shape
        cent = (int(col / 2), int(row / 2))
        max_radius = int(np.sqrt(row**2 + col**2) / 2)
        return cv2.linearPolar(img, cent, max_radius, flag)

    def _filter_sti(self, sti: np.ndarray) -> NDArray:
        """Filter image.

        Using method proposed in:
        "An improvement of the Space-Time Image Velocimetry combined with a new
        denoising method for estimating river discharge"
        by:
        - Zhao, Haoyuan
        - Chen, Hua
        - Liu, Bingyi
        - Liu, Weigao
        - Xu, Chong Yu
        - Guo, Shenglian
        - Wang, Jun
        """
        # crop and resize in order to have more precision
        x = min(sti.shape)
        sti = sti[:, :x] if x == sti.shape[0] else sti[:x, :]
        LOG.debug(f"size before reshape: {x}")
        # the example of the paper uses 600x600, so do I
        sti = cv2.resize(sti, (600, 600), interpolation=cv2.INTER_LINEAR)
        self.save(f"f_{self.cnt}_0.npy", sti)

        # WINDOW FUNCTION FILTERING
        size = sti.shape
        # TODO: Use a better 2d convolution function
        sti = self._conv2d(sti, self._filter_win)
        self.save(f"f_{self.cnt}_1.npy", sti)

        # DETECTION OF PRINCIPAL DIRECTION OF FOURIER SPECTRUM
        sti_ft = np.abs(np.fft.fftshift(np.fft.fft2(sti)))
        # filter vertical and horizontal patterns
        c_x = int(sti_ft.shape[0] / 2)
        c_y = int(sti_ft.shape[1] / 2)
        sti_ft[c_x - self._vh_filter : c_x + self._vh_filter, :] = 0
        sti_ft[:, c_y - self._vh_filter : c_y + self._vh_filter] = 0
        self.save(f"f_{self.cnt}_2.npy", sti_ft)
        # transform to polar system
        sti_ft_polar = self._to_polar_system(sti_ft)
        self.save(f"f_{self.cnt}_3.npy", sti_ft_polar)

        # FILTER IN FREQUENCY DOMAIN
        polar_mask = self._generate_polar_mask(sti_ft_polar)
        sti_ft_polar = sti_ft_polar * polar_mask
        self.save(f"f_{self.cnt}_4.npy", sti_ft_polar)

        sti_ft_filtered = self._to_polar_system(sti_ft_polar, "invert")
        self.save(f"f_{self.cnt}_5.npy", sti_ft_filtered)

        sti_filtered = np.abs(np.fft.ifft2(np.fft.ifftshift(sti_ft_filtered)))
        self.save(f"f_{self.cnt}_6.npy", sti_filtered)

        sti_filtered = cv2.resize(
            sti_filtered, (size[1], size[0]), interpolation=cv2.INTER_AREA
        )

        return np.interp(
            sti_filtered, (sti_filtered.min(), sti_filtered.max()), (0, 255)
        ).astype(np.uint8)

    def _generate_final_image(self, sti: NDArray, mask: NDArray) -> NDArray:
        """Generate rgb image."""
        new_sti = np.interp(sti, (sti.min(), sti.max()), (0, 255)).astype(
            np.uint8
        )
        new_sti = cv2.equalizeHist(sti)
        new_sti = cv2.cvtColor(new_sti, cv2.COLOR_GRAY2RGB)
        new_mask = np.interp(mask, (mask.min(), mask.max()), (0, 255)).astype(
            np.uint8
        )

        new_mask = cv2.cvtColor(new_mask, cv2.COLOR_GRAY2RGB)
        return cv2.add(new_sti, new_mask)

    @staticmethod
    def _squarify(m: NDArray) -> NDArray:
        (a, b) = m.shape
        padding = ((0, 0), (0, a - b)) if a > b else ((0, b - a), (0, 0))
        return np.pad(m, padding)

    def _calculate_mot_using_fft(self, sti: NDArray) -> tuple[float, NDArray]:
        """Calcualte MOT using FFT."""
        self.save(f"g_{self.cnt}_0.npy", sti)
        sti_canny = cv2.Canny(sti, 10, 10)
        self.save(f"g_{self.cnt}_1.npy", sti_canny)
        sti_padd = self._squarify(sti_canny)
        self.save(f"g_{self.cnt}_2.npy", sti_padd)
        sti_ft = np.abs(np.fft.fftshift(np.fft.fft2(sti_padd)))
        self.save(f"g_{self.cnt}_3.npy", sti_ft)
        sti_ft_polar = self._to_polar_system(sti_ft)
        self.save(f"g_{self.cnt}_4.npy", sti_ft_polar)
        isd = np.sum(sti_ft_polar.T, axis=0)
        freq, _ = self._get_main_freqs(isd)
        angle0 = 2 * math.pi * freq / sti_ft_polar.shape[0]
        angle1 = 2 * math.pi * freq / sti_ft_polar.shape[1]
        angle = (angle0 + angle1) / 2
        velocity = self.get_velocity(angle)
        LOG.debug(f"angle: {angle:.2f}")
        LOG.debug(f"velocity: {velocity:.2f}")
        mask = np.zeros(sti.shape)
        mask = self._draw_angle(
            mask,
            angle,
            (int(sti.shape[1] / 2), int(sti.shape[0] / 2)),
            thick=10,
            amplitud=80,
        )

        return velocity, mask

    def _calculate_mot_using_gmt(self, sti: NDArray) -> tuple[float, NDArray]:
        """Calcualte MOT using GMT.

        explained:
        "Development of a non-intrusive and efficient flow monitoring
        technique: The space-time image velocimetry (STIV)"
        """
        window_width = int(self.config.window_shape[0] / 2)
        window_height = int(self.config.window_shape[1] / 2)

        width = sti.shape[0]
        height = sti.shape[1]

        angle_accumulated = 0
        c_total = 0

        # plot vectors
        mask = np.zeros(sti.shape)

        s = window_width
        i = 0
        while s + window_width < width:
            j = 0
            e = window_height
            while e + window_height < height:
                ss = slice(s - window_width, s + window_width)
                ee = slice(e - window_height, e + window_height)
                image_window = sti[ss, ee]
                angle, coherence = self._process_sti(image_window)
                angle_accumulated += angle * coherence
                c_total += coherence
                LOG.debug(
                    f"- at ({i}, {j}): angle = "
                    f"- in ({s}, {e}): angle = "
                    f"{math.degrees(angle):0.2f}, "
                    f"coherence={coherence:0.2f}, "
                    f"velocity={round(self.get_velocity(angle), 2)}"
                )
                mask = self._draw_angle(mask, angle, (e, s))
                j += 1
                e += int(self.config.overlap)
            i += 1
            s += int(self.config.overlap)

        mean_angle = angle_accumulated / c_total

        velocity = self.get_velocity(mean_angle)
        LOG.debug(f"weighted mean angle: {round(math.degrees(mean_angle), 2)}")
        LOG.debug(f"velocity {round(velocity, 2)}")

        return velocity, mask

    def run(self, show_image: bool = False) -> dict[str, dict[str, float]]:
        """Execute."""
        velocities = []
        for idx, sti_ in enumerate(self.stis):
            LOG.debug(f"space time image {idx} shape: {sti_.shape}")
            self.cnt = idx
            sti = self._filter_sti(sti_)
            # velocity, mask = self._calculate_MOT_using_GMT(sti)
            velocity, mask = self._calculate_mot_using_fft(sti)
            velocities.append(velocity)

            final_image: NDArray = sti + mask
            self.save(f"stiv_final_{idx:02}.npy", final_image)
            cv2.imwrite(
                f"images/stiv/stiv_final_{idx:02}.png",
                self._generate_final_image(sti, mask),
            )
            if show_image:
                cv2.imshow("stiv final", final_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        total = 0
        out_json = {}
        for i, vel in enumerate(velocities):
            total += vel
            out_json[str(i)] = {}
            out_json[str(i)]["velocity"] = vel
        total /= len(velocities)
        LOG.debug(f"Total mean velocity: {round(total, 2)}")
        return out_json


def main(
    config_fp: Path,
    show_image: bool = False,
    debug: bool = False,
    images_dp: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Execute example of STIV usage."""
    logging.basicConfig(
        level=logging.INFO if not debug else logging.DEBUG,
        format="%(asctime)s | %(levelname).1s | %(name).20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    t0 = time.process_time()
    config = Config.from_fp(config_fp)
    if config.stiv is None:
        raise ValueError("STIV configuration not found")
    loader: Loader = make_loader(config.dataset)
    formatter = Formatter(config.dataset, config.preprocessing)
    stiv = STIV(config.stiv, loader, formatter, config.lines, images_dp)
    t1 = time.process_time()
    ret = stiv.run(show_image)
    t2 = time.process_time()
    print(ret)
    LOG.info(f"STIV {t1 - t0}")
    LOG.info(f"stiv.run {t2 - t1}")
    return ret


if __name__ == "__main__":
    import typer

    typer.run(main)
