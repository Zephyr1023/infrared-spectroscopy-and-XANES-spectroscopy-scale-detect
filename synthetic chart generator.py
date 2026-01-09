import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
import random
import json
import os
import gc
import uuid
import string
from PIL import Image, ImageFilter

# --- Configuration ---
OUTPUT_IMG_DIR = "dataset_v8_final/synthetic images"
OUTPUT_JSON_DIR = "dataset_v8_final/synthetic labels_json"
OUTPUT_TXT_DIR = "dataset_v8_final/synthetic labels_txt"
OUTPUT_POSE_DIR = "dataset_v8_final/synthetic labels_pose"

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)
os.makedirs(OUTPUT_POSE_DIR, exist_ok=True)


class DataGenerator:
    """Generates synthetic data distributions."""

    @staticmethod
    def generate(chart_type, num_points, magnitude):
        if chart_type == 'pie':
            return [random.randint(10, 100) for _ in range(random.randint(3, 7))]

        trend = random.choice(['linear', 'exponential', 'sigmoid', 'sine', 'step', 'random', 'mixed'])
        x = np.linspace(0, 10, num_points)
        noise = np.random.normal(0, magnitude * 0.05, num_points)

        if trend == 'linear':
            slope = random.uniform(-1, 1) * magnitude
            intercept = random.uniform(0, magnitude)
            y = slope * x + intercept + noise
        elif trend == 'exponential':
            base = random.uniform(0, 5)
            y = np.exp(x / 4) * (magnitude / 15) + base + noise
        elif trend == 'sigmoid':
            y = magnitude / (1 + np.exp(-1 * (x - 5))) + noise
        elif trend == 'step':
            steps = random.randint(3, 8)
            y = np.zeros_like(x)
            indices = np.array_split(np.arange(num_points), steps)
            val = random.uniform(0, magnitude)
            for idx in indices:
                y[idx] = val
                val += random.uniform(-magnitude / 5, magnitude / 5)
            y += noise
        elif trend == 'sine':
            freq = random.uniform(0.2, 5.0)
            y = np.sin(x * freq) * magnitude + magnitude / 2 + noise
        elif trend == 'mixed':
            y1 = np.sin(x * random.uniform(0.5, 2.0)) * magnitude
            y2 = random.uniform(-1, 1) * magnitude * x / 10
            y = y1 + y2 + magnitude + noise
        else:
            y = np.random.rand(num_points) * magnitude

        if chart_type in ['bar']:
            y = np.abs(y)

        return x, y

    @staticmethod
    def random_text(length=8, with_number=True, with_dot=True):
        chars = string.ascii_letters + ' '
        if with_number: chars += string.digits
        if with_dot: chars += '.'
        if length < 1: length = 1
        return ''.join(random.choices(chars, k=random.randint(1, length))).strip()

    @staticmethod
    def random_fig_label():
        pattern = random.choice(['fig_num', 'letter', 'paren_letter'])
        if pattern == 'fig_num':
            return f"Fig. {random.randint(1, 20)}"
        elif pattern == 'letter':
            return random.choice(string.ascii_lowercase)
        elif pattern == 'paren_letter':
            return f"({random.choice(string.ascii_lowercase)})"
        return "a"


class PrecisionChartGenerator:
    def __init__(self):
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 'black']
        self.line_styles = ['-', '--', '-.', ':']
        self.markers = ['o', 's', '^', 'D', 'x', '*', '+', '.', 'v', '<', '>', 'p', 'h']
        self.font_families = ['sans-serif', 'serif', 'monospace', 'cursive', 'fantasy']
        self.units = ['%', 'kg', 'm', 's', '$', '°C', 'mm', 'Hz', 'V', 'A', 'W', 'J', 'Pa']
        self.title_words = ["Analysis", "Report", "Growth", "Performance", "Data", "Trends", "Comparison",
                            "Study", "Experiment", "Result", "Distribution", "Frequency"]
        self.annotation_words = ["Peak", "Max", "Min", "Valley", "Note", "Ref.", "Point", "Outlier"]

    def add_noise(self, img):
        if random.random() > 0.85: return img
        mode = random.choice(['blur', 'salt', 'downsample', 'jpeg', 'noise'])
        if mode == 'blur':
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        elif mode == 'downsample':
            w, h = img.size
            f = random.uniform(0.5, 0.8)
            img = img.resize((int(w * f), int(h * f)), Image.Resampling.BILINEAR)
            img = img.resize((w, h), Image.Resampling.NEAREST)
        elif mode == 'salt':
            arr = np.array(img)
            mask = np.random.random(arr.shape[:2])
            arr[mask < 0.005] = 0
            arr[mask > 0.995] = 255
            img = Image.fromarray(arr)
        elif mode == 'jpeg':
            import io
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=random.randint(30, 80))
            buffer.seek(0)
            img = Image.open(buffer)
        elif mode == 'noise':
            arr = np.array(img).astype(np.float32)
            noise_img = arr + np.random.normal(0, random.randint(5, 20), arr.shape)
            noise_img = np.clip(noise_img, 0, 255).astype(np.uint8)
            img = Image.fromarray(noise_img)
        return img

    def get_mpl_bbox(self, artist, renderer, img_h, padding=0):
        if not artist or not artist.get_visible(): return None
        try:
            bbox = artist.get_window_extent(renderer)
            if bbox.width <= 0 or bbox.height <= 0: return None
            x_min = bbox.x0 - padding
            y_min = img_h - bbox.y1 - padding
            w = bbox.width + (padding * 2)
            h = bbox.height + (padding * 2)
            return [round(x_min, 2), round(y_min, 2), round(w, 2), round(h, 2)]
        except:
            return None

    def extract_axis_pairs(self, axis, renderer, img_h, ax_bbox):
        pairs = []
        view_min, view_max = sorted(axis.get_view_interval())

        try:
            ticks = axis.get_major_ticks()
        except:
            return []

        for t in ticks:
            try:
                loc = t.get_loc()
                if not (view_min <= loc <= view_max): continue
            except:
                pass

            mark_bbox = self.get_mpl_bbox(t.tick1line, renderer, img_h, padding=2.0)
            if not mark_bbox: mark_bbox = self.get_mpl_bbox(t.tick2line, renderer, img_h, padding=2.0)

            label_bbox = self.get_mpl_bbox(t.label1, renderer, img_h, padding=1.5)
            if not label_bbox: label_bbox = self.get_mpl_bbox(t.label2, renderer, img_h, padding=1.5)

            is_valid = False
            tolerance = 80  # Increased tolerance to catch labels slightly outside

            if mark_bbox:
                cx, cy = mark_bbox[0] + mark_bbox[2] / 2, mark_bbox[1] + mark_bbox[3] / 2
                # Broad check
                is_valid = True

            if label_bbox:
                cx, cy = label_bbox[0] + label_bbox[2] / 2, label_bbox[1] + label_bbox[3] / 2
                is_valid = True

            if is_valid and (mark_bbox or label_bbox):
                txt = t.label1.get_text() if t.label1.get_text() else t.label2.get_text()
                pairs.append({"text": txt, "mark_bbox": mark_bbox, "label_bbox": label_bbox})

        return pairs

    def apply_random_styles(self, fig, ax):
        font_family = random.choice(self.font_families)
        plt.rcParams['font.family'] = font_family
        plt.rcParams['font.size'] = random.randint(9, 14)

        spine_width = random.uniform(0.5, 3.0)
        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        tick_width = random.uniform(0.5, 3.0)
        tick_length = random.uniform(2.0, 8.0)
        ax.tick_params(axis='both', which='major', width=tick_width, length=tick_length)
        ax.tick_params(axis='both', which='minor', width=tick_width * 0.7, length=tick_length * 0.6)

        # Background color
        if random.random() > 0.3:
            bg_color = (random.uniform(0.85, 1), random.uniform(0.85, 1), random.uniform(0.85, 1))
            ax.set_facecolor(bg_color)
        else:
            ax.set_facecolor('white')

        # Grid
        if random.random() > 0.4:
            grid_lw = random.uniform(0.5, 2.0)
            ax.grid(True, linestyle=random.choice(['-', '--', ':', '-.']),
                    alpha=random.uniform(0.3, 0.8),
                    linewidth=grid_lw,
                    color=random.choice(['gray', 'lightgray', 'black', 'blue']))
            ax.set_axisbelow(True)

    def apply_tick_anomalies(self, ax, chart_type):
        if chart_type == 'pie': return

        y_min, y_max = ax.get_ylim()

        # Log Scale
        if random.random() < 0.15 and chart_type != 'bar' and y_min > 0:
            try:
                if random.random() > 0.5:
                    ax.set_yscale('log')
                else:
                    ax.set_xscale('log')
            except:
                ax.set_yscale('linear')
                ax.set_xscale('linear')

        # Unit Formatter
        if random.random() < 0.25:
            unit = random.choice(self.units)

            def unit_formatter(x, pos):
                return f'{x:.0f}{unit}'

            if random.random() > 0.5:
                ax.yaxis.set_major_formatter(ticker.FuncFormatter(unit_formatter))
            else:
                ax.xaxis.set_major_formatter(ticker.FuncFormatter(unit_formatter))

        # MaxNLocator (Tick density)
        if random.random() < 0.2:
            nbins = random.randint(5, 15)
            if random.random() > 0.5:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))
                ax.tick_params(axis='x', rotation=random.choice([30, 45, 60, 90]))
            else:
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbins))

    def add_chart_clutter(self, ax, x_data, y_data):
        if random.random() > 0.3: return
        if x_data is None or y_data is None or len(x_data) < 5: return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        y_range = ylim[1] - ylim[0]
        x_range = xlim[1] - xlim[0]

        num_arrows = random.randint(2, 6)  # Reduced slightly
        indices = random.sample(range(len(x_data)), min(num_arrows, len(x_data)))

        for idx in indices:
            target_x = x_data[idx]
            target_y = y_data[idx]
            # Ensure target is within limits
            if not (ylim[0] <= target_y <= ylim[1]): continue

            y_offset = random.uniform(0.08, 0.20) * y_range
            x_offset = random.uniform(-0.03, 0.03) * x_range
            text_x = target_x + x_offset
            text_y = target_y + y_offset
            if text_y > ylim[1]:
                text_y = ylim[1] - (y_offset * 0.1)

            annot_type = random.choice(['number', 'word', 'short_code'])
            if annot_type == 'number':
                label = str(random.randint(1, 150))
            elif annot_type == 'word':
                label = random.choice(self.annotation_words)
            else:
                label = DataGenerator.random_text(3, with_number=True, with_dot=False).upper()

            color = random.choice(['black', 'red', 'blue', 'darkgray'])
            arrow_style = random.choice(['->', '-|>'])
            arrow_lw = random.uniform(0.8, 2.0)

            ax.annotate(label,
                        xy=(target_x, target_y),
                        xytext=(text_x, text_y),
                        xycoords='data',
                        textcoords='data',
                        arrowprops=dict(arrowstyle=arrow_style, color=color, lw=arrow_lw,
                                        shrinkB=5,
                                        connectionstyle="arc3,rad={}".format(random.uniform(-0.1, 0.1))),
                        fontsize=random.randint(8, 11), color=color,
                        ha='center', va='bottom')

    def apply_legend_variations(self, ax, chart_type='generic'):
        loc_type = random.choice(['inside', 'outside_right', 'outside_bottom', 'inside_corner'])
        fontsize = random.randint(10, 16)
        title = DataGenerator.random_text(length=6) if random.random() > 0.7 else None
        frameon = random.choice([True, False])

        try:
            if loc_type == 'inside':
                ax.legend(loc='best', fontsize=fontsize, title=title, frameon=frameon)
            elif loc_type == 'inside_corner':
                loc = random.choice(['upper left', 'upper right', 'lower left', 'lower right'])
                ax.legend(loc=loc, fontsize=fontsize, title=title, frameon=frameon)
            elif loc_type == 'outside_right':
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=fontsize, title=title)
            elif loc_type == 'outside_bottom':
                ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=random.randint(2, 4), fontsize=fontsize,
                          title=title)
        except:
            pass  # Handle cases where no labels exist

    def apply_axis_augmentation(self, ax):
        x_loc = random.choice(['bottom', 'top'])
        y_loc = random.choice(['left', 'right'])
        ax.xaxis.set_ticks_position(x_loc)
        ax.xaxis.set_label_position(x_loc)
        ax.yaxis.set_ticks_position(y_loc)
        ax.yaxis.set_label_position(y_loc)

        if random.random() > 0.6:
            offset = random.randint(5, 20)
            ax.spines[x_loc].set_position(('outward', offset))
            ax.spines[y_loc].set_position(('outward', offset))
            for spine in ax.spines:
                if spine not in [x_loc, y_loc]: ax.spines[spine].set_visible(False)

        if random.random() > 0.8:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    def add_figure_numbering(self, fig, axes_list):
        label_text = DataGenerator.random_fig_label()
        strategy = random.choice(['canvas_corner', 'axis_near', 'plot_inner'])
        fontsize = random.randint(18, 26)
        fontweight = 'bold'
        color = 'black'

        if strategy == 'canvas_corner':
            padding = 0.02
            corners = [
                (padding, 1 - padding, 'left', 'top'),
                (1 - padding, 1 - padding, 'right', 'top'),
                (padding, padding, 'left', 'bottom'),
                (1 - padding, padding, 'right', 'bottom')
            ]
            x, y, ha, va = random.choice(corners)
            fig.text(x, y, label_text, transform=fig.transFigure,
                     fontsize=fontsize, fontweight=fontweight, color=color, ha=ha, va=va)
        elif strategy == 'axis_near' and len(axes_list) > 0:
            target_ax = random.choice(axes_list)
            if random.random() > 0.5:
                target_ax.text(0.5, random.uniform(-0.3, -0.2), label_text, transform=target_ax.transAxes,
                               fontsize=fontsize, fontweight=fontweight, color=color, ha='center', va='top')
            else:
                target_ax.text(random.uniform(-0.3, -0.2), 0.5, label_text, transform=target_ax.transAxes,
                               fontsize=fontsize, fontweight=fontweight, color=color, ha='right', va='center')
        else:
            fig.text(0.02, 0.98, label_text, transform=fig.transFigure,
                     fontsize=fontsize, fontweight=fontweight, color=color, ha='left', va='top')

    def create_one_sample(self, file_id):
        layout_choice = random.choices([(1, 1), (1, 2), (2, 1), (2, 2)], weights=[0.6, 0.15, 0.15, 0.1])[0]
        rows, cols = layout_choice
        base_w, base_h = random.randint(8, 14), random.randint(7, 10)
        total_w = base_w * cols
        total_h = base_h * rows
        dpi = random.choice([80, 100, 120])

        fig, axes = plt.subplots(rows, cols, figsize=(total_w, total_h), dpi=dpi)
        if rows == 1 and cols == 1:
            axes_flat = [axes]
        else:
            axes_flat = axes.flatten()

        charts_meta = []

        for idx, ax in enumerate(axes_flat):
            chart_type = random.choices(['line', 'bar', 'scatter', 'pie'], weights=[0.5, 0.2, 0.2, 0.1])[0]
            self.apply_random_styles(fig, ax)

            magnitude = 10 ** random.randint(1, 5)
            num_points = random.randint(15, 80)

            # --- DUAL Y-AXIS LOGIC START ---
            is_dual_axis = False
            ax2 = None
            if chart_type in ['line', 'bar'] and random.random() < 0.30:  # 30% chance for dual axis
                is_dual_axis = True
                ax2 = ax.twinx()
                # Apply similar style augmentations to the second axis
                self.apply_tick_anomalies(ax2, chart_type)
            # --- DUAL Y-AXIS LOGIC END ---

            if chart_type == 'pie':
                data = DataGenerator.generate('pie', 0, 0)
                autopct = random.choice(['%1.1f%%', '%1.0f%%', None])
                pie_labels = [DataGenerator.random_text(3) for _ in range(len(data))]
                pie_returns = ax.pie(data, labels=pie_labels, autopct=autopct, startangle=random.randint(0, 360))
                wedges = pie_returns[0]

                title = f"{random.choice(self.title_words)} {DataGenerator.random_text(3)}"
                ax.set_title(title, fontsize=random.randint(16, 28))
                if random.choice([True, False]):
                    ax.legend(wedges, pie_labels, title="Legend", loc="best")
                else:
                    ax.legend(wedges, pie_labels, title="Legend", bbox_to_anchor=(1, 0, 0.5, 1))
                xl_obj, yl_obj = None, None

            else:
                # Primary Axis Data
                num_series = random.randint(1, 4) if is_dual_axis else random.randint(1, 6)
                x_data, _ = DataGenerator.generate('linear', num_points, 10)
                colors = random.choices(self.colors, k=num_series + 2)  # Extra colors for dual axis
                styles = random.choices(self.line_styles, k=num_series + 2)
                markers = random.choices(self.markers, k=num_series + 2)

                # Plot on Primary Axis (Left)
                for s_i in range(num_series):
                    _, y_data = DataGenerator.generate(chart_type, num_points, magnitude)
                    label = f"{DataGenerator.random_text(4, with_dot=False)} {s_i + 1}"

                    if chart_type == 'line':
                        lw = random.uniform(0.5, 5.0)
                        alpha = random.uniform(0.6, 1.0)
                        marker = markers[s_i] if random.random() > 0.5 else None
                        ax.plot(x_data, y_data, label=label, color=colors[s_i],
                                linestyle=styles[s_i], linewidth=lw, alpha=alpha, marker=marker)
                    elif chart_type == 'bar':
                        # Shift bars slightly if dual axis to avoid overlap is tricky, simple overlap for now
                        width = 0.8 / num_series
                        ax.bar(np.arange(num_points) + s_i * width, y_data, width=width,
                               label=label, color=colors[s_i], alpha=random.uniform(0.7, 1.0))
                    elif chart_type == 'scatter':
                        ax.scatter(x_data, y_data, label=label, marker=markers[s_i],
                                   color=colors[s_i], s=random.randint(20, 100), alpha=random.uniform(0.5, 0.9))

                # Plot on Secondary Axis (Right) if active
                if is_dual_axis and ax2:
                    # Use a different magnitude for the second axis to ensure scales are different
                    magnitude2 = magnitude * random.choice([0.01, 0.1, 10, 50])
                    num_series2 = random.randint(1, 2)

                    for s_j in range(num_series2):
                        _, y_data2 = DataGenerator.generate(chart_type, num_points, magnitude2)
                        label2 = f"Right {DataGenerator.random_text(3)} {s_j + 1}"
                        color2 = colors[num_series + s_j]  # Use different colors

                        if chart_type == 'line':
                            ax2.plot(x_data, y_data2, label=label2, color=color2,
                                     linestyle=random.choice(self.line_styles), linewidth=random.uniform(1, 4))
                        elif chart_type == 'bar':
                            # Plot as line on dual axis usually looks better mixed, or bar with transparency
                            ax2.plot(x_data, y_data2, label=label2, color=color2, marker='x', linestyle='--')

                # Augmentations
                self.apply_tick_anomalies(ax, chart_type)
                if not is_dual_axis:
                    self.apply_axis_augmentation(ax)  # Only flip axis positions if single axis

                self.apply_legend_variations(ax, chart_type)

                xl_text = f"{DataGenerator.random_text(length=15)} ({DataGenerator.random_text(3)})"
                yl_text = f"{DataGenerator.random_text(length=15)} ({DataGenerator.random_text(3)})"

                xl_obj = ax.set_xlabel(xl_text, fontsize=random.randint(10, 14))
                yl_obj = ax.set_ylabel(yl_text, fontsize=random.randint(10, 14))

                if is_dual_axis and ax2:
                    yl_text2 = f"{DataGenerator.random_text(length=10)} (Right)"
                    ax2.set_ylabel(yl_text2, fontsize=random.randint(10, 14), color=colors[-1])
                    ax2.tick_params(axis='y', labelcolor=colors[-1])

                title = f"{random.choice(self.title_words)}: {DataGenerator.random_text(10)}"
                ax.set_title(title, fontsize=random.randint(20, 28))

                self.add_chart_clutter(ax, x_data, y_data)

            charts_meta.append({"ax": ax, "ax2": ax2, "type": chart_type, "xl_obj": xl_obj, "yl_obj": yl_obj})

        plt.tight_layout(pad=random.uniform(2.0, 4.0))
        self.add_figure_numbering(fig, axes_flat)

        fig.canvas.draw()
        actual_w, actual_h = fig.canvas.get_width_height()
        renderer = fig.canvas.get_renderer()

        final_charts_data = []
        for meta in charts_meta:
            ax = meta['ax']
            ax2 = meta.get('ax2')

            # Use the primary axis bbox for the whole chart area
            ax_bbox = ax.get_window_extent()

            chart_data = {
                "type": meta['type'],
                "bbox": [ax_bbox.x0, actual_h - ax_bbox.y1, ax_bbox.width, ax_bbox.height],
                "x_axis": {"ticks": []},
                "y_axis": {"ticks": []}
            }
            if meta['xl_obj']:
                chart_data['x_axis']['title_bbox'] = self.get_mpl_bbox(meta['xl_obj'], renderer, actual_h)
            if meta['yl_obj']:
                chart_data['y_axis']['title_bbox'] = self.get_mpl_bbox(meta['yl_obj'], renderer, actual_h)

            if meta['type'] != 'pie':
                # X Axis
                chart_data['x_axis']['ticks'] = self.extract_axis_pairs(ax.xaxis, renderer, actual_h, ax_bbox)

                # Y Axis (Left)
                left_ticks = self.extract_axis_pairs(ax.yaxis, renderer, actual_h, ax_bbox)
                chart_data['y_axis']['ticks'].extend(left_ticks)

                # Y Axis (Right) - If exists
                if ax2:
                    # Note: We append right-axis ticks to the SAME 'y_axis' list.
                    # YOLO will treat them all as class 1.
                    right_ticks = self.extract_axis_pairs(ax2.yaxis, renderer, actual_h, ax_bbox)
                    chart_data['y_axis']['ticks'].extend(right_ticks)

            final_charts_data.append(chart_data)

        buf = fig.canvas.buffer_rgba()
        final_img = Image.fromarray(np.asarray(buf)).convert("RGB")
        final_img = self.add_noise(final_img)
        fname = f"{file_id}.jpg"
        final_img.save(os.path.join(OUTPUT_IMG_DIR, fname), quality=random.randint(85, 95))

        json_data = {
            "image": fname,
            "width": actual_w,
            "height": actual_h,
            "layout": f"{rows}x{cols}",
            "charts": final_charts_data
        }

        with open(os.path.join(OUTPUT_JSON_DIR, f"{file_id}.json"), 'w') as f:
            json.dump(json_data, f, indent=2)

        self.save_yolo_txt(file_id, json_data)
        self.save_yolo_pose(file_id, json_data)

        plt.close(fig)

    def save_yolo_txt(self, file_id, data):
        yolo_lines = []
        w, h = data['width'], data['height']
        for chart in data['charts']:
            if 'x_axis' in chart:
                for item in chart['x_axis']['ticks']:
                    if item['mark_bbox']:
                        bx, by, bw, bh = item['mark_bbox']
                        yolo_lines.append(
                            f"0 {(bx + bw / 2) / w:.6f} {(by + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                    if item['label_bbox']:
                        bx, by, bw, bh = item['label_bbox']
                        yolo_lines.append(
                            f"1 {(bx + bw / 2) / w:.6f} {(by + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            if 'y_axis' in chart:
                for item in chart['y_axis']['ticks']:
                    if item['mark_bbox']:
                        bx, by, bw, bh = item['mark_bbox']
                        yolo_lines.append(
                            f"0 {(bx + bw / 2) / w:.6f} {(by + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
                    if item['label_bbox']:
                        bx, by, bw, bh = item['label_bbox']
                        yolo_lines.append(
                            f"1 {(bx + bw / 2) / w:.6f} {(by + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
        with open(os.path.join(OUTPUT_TXT_DIR, f"{file_id}.txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))

    def save_yolo_pose(self, file_id, data):
        yolo_lines = []
        w_img, h_img = data['width'], data['height']
        for chart in data['charts']:
            def process_axis_pose(axis_data, class_id):
                for item in axis_data['ticks']:
                    if item['mark_bbox'] and item['label_bbox']:
                        mb, lb = item['mark_bbox'], item['label_bbox']
                        mx, my = mb[0] + mb[2] / 2, mb[1] + mb[3] / 2
                        lx, ly = lb[0] + lb[2] / 2, lb[1] + lb[3] / 2
                        x_min, y_min = min(mb[0], lb[0]), min(mb[1], lb[1])
                        x_max, y_max = max(mb[0] + mb[2], lb[0] + lb[2]), max(mb[1] + mb[3], lb[1] + lb[3])
                        box_w, box_h = x_max - x_min, y_max - y_min
                        box_xc, box_yc = x_min + box_w / 2, y_min + box_h / 2

                        line = (
                            f"{class_id} {box_xc / w_img:.6f} {box_yc / h_img:.6f} {box_w / w_img:.6f} {box_h / h_img:.6f} "
                            f"{mx / w_img:.6f} {my / h_img:.6f} 2 {lx / w_img:.6f} {ly / h_img:.6f} 2")
                        yolo_lines.append(line)

            if 'x_axis' in chart: process_axis_pose(chart['x_axis'], 0)
            if 'y_axis' in chart: process_axis_pose(chart['y_axis'], 1)
        with open(os.path.join(OUTPUT_POSE_DIR, f"{file_id}.txt"), 'w') as f:
            f.write('\n'.join(yolo_lines))

    def generate_batch(self, count):
        print(f"Generating {count} charts with Dual-Axis Capability...")
        import matplotlib
        matplotlib.use('Agg')

        for i in range(count):
            uid = str(uuid.uuid4())[:8]
            try:
                self.create_one_sample(uid)
                plt.close('all')
                if (i + 1) % 50 == 0:
                    gc.collect()
                    print(f"Progress: {i + 1}/{count} (GC collected)")
                elif (i + 1) % 10 == 0:
                    print(f"Progress: {i + 1}/{count}")

            except MemoryError:
                print(f"Skipping {uid} due to MemoryError (RAM full).")
                plt.close('all')
                gc.collect()
            except Exception as e:
                print(f"Error {uid}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    random.seed(None)
    np.random.seed(None)

    gen = PrecisionChartGenerator()
    gen.generate_batch(4900)  # Adjust this number as needed
    print(f"Done! Check {OUTPUT_IMG_DIR}")