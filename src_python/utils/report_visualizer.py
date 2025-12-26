import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportVisualizer:
    """
    A tool to convert Pipeline JSON reports into Pandas DataFrames and 
    generate human-readable graphs with physical units (nm) and statistics.
    """
    def __init__(self, config_dict, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else None
        self.cfg = config_dict
        
        # Extract pixel size for conversion (Default to 1.0 if missing)
        # Try common keys: 'pixel_size_nm', or inside 'microscope_parameters'
        self.px_to_nm = self._get_pixel_size()
        logger.info(f"ReportVisualizer initialized with Pixel Size: {self.px_to_nm} nm/px")

        self.style_cfg = {
            'hist_bins': 30,
            'color_ne': 'skyblue',
            'color_fov': 'salmon',
            'color_fit': 'black',
            'alpha': 0.7
        }

    def _get_pixel_size(self):
        """Helper to find pixel size in the config."""
        # 1. Direct key
        if 'pixel_size_nm' in self.cfg:
            return float(self.cfg['pixel_size_nm'])
        
        # 2. Nested in microscope_parameters
        if 'microscope_parameters' in self.cfg:
            return float(self.cfg['microscope_parameters'].get('pixel_size_nm', 1.0))
            
        # 3. Fallback (Warn user)
        logger.warning("No 'pixel_size_nm' found in config. Using 1.0 (Output will be in pixels).")
        return 1.0

    # --- LOADER & PARSER METHODS ---

    def load_json(self, filepath):
        path = Path(filepath)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return None
        with open(path, 'r') as f:
            return json.load(f)

    def flatten_ne_stability_report(self, json_data):
        if not json_data: return pd.DataFrame()
        rows = []
        for fov_id, nuclei in json_data.items():
            for label, data in nuclei.items():
                metrics = data.get('metrics', {})
                row = {
                    'fov_id': fov_id,
                    'ne_label': label,
                    'status': data.get('status'),
                    'drift_px': metrics.get('drift'),
                    'drift_dx_px': metrics.get('drift_dx'), 
                    'drift_dy_px': metrics.get('drift_dy'),
                    'sigma_px': metrics.get('sigma_combined')
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        # Convert to NM
        if not df.empty:
            df['drift_nm'] = df['drift_px'] * self.px_to_nm
            df['drift_dx_nm'] = df['drift_dx_px'] * self.px_to_nm
            df['drift_dy_nm'] = df['drift_dy_px'] * self.px_to_nm
            df['sigma_nm'] = df['sigma_px'] * self.px_to_nm
        return df

    def flatten_fov_stability_report(self, json_data):
        if not json_data: return pd.DataFrame()
        rows = []
        for fov_id, data in json_data.items():
            row = {
                'fov_id': fov_id,
                'drift_px': data.get('drift'),
                'drift_dx_px': data.get('drift_dx'),
                'drift_dy_px': data.get('drift_dy'),
                'status': data.get('status')
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        if not df.empty:
            df['drift_nm'] = df['drift_px'] * self.px_to_nm
            df['drift_dx_nm'] = df['drift_dx_px'] * self.px_to_nm
            df['drift_dy_nm'] = df['drift_dy_px'] * self.px_to_nm
        return df

    def flatten_raw_registration(self, json_data):
        if not json_data: return pd.DataFrame()
        rows = []
        for fov_id, content in json_data.items():
            if not isinstance(content, dict): continue
            ne_keys = [k for k in content.keys() if isinstance(content[k], dict) and 'shift_vector' in content[k]]
            for label in ne_keys:
                d = content[label]
                row = {
                    'fov_id': fov_id,
                    'ne_label': label,
                    'angle_deg': np.degrees(d.get('angle', 0.0)),
                    'scale_factor': d.get('scale', 1.0),
                    'shift_mag_px': np.linalg.norm(d['shift_vector'])
                }
                rows.append(row)
        return pd.DataFrame(rows)

    # --- PLOTTING METHODS ---

    def plot_histogram_with_fit(self, df, column, title, xlabel, color='skyblue', save_name=None, return_fig=False):
        """
        Plots histogram with Gaussian fit AND a stats box.
        """
        if df.empty or column not in df.columns:
            return None

        # Clean Data
        data = pd.to_numeric(df[column], errors='coerce').dropna()
        data = data[np.isfinite(data)]
        
        if len(data) < 2:
            return None

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6)) # Wider for stats box
        
        # 1. Histogram
        try:
            count, bins, ignored = ax.hist(data, bins=self.style_cfg['hist_bins'], 
                                            density=True, alpha=self.style_cfg['alpha'], 
                                            color=color, edgecolor='black', label='Data')
            
            # 2. Gaussian Fit
            mu, std = norm.fit(data)
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            ax.plot(x, p, 'k', linewidth=2, label=r'Fit: $\mu={:.2f}, \sigma={:.2f}$'.format(mu, std))
            
            # 3. Descriptive Statistics Box
            stats_text = (
                f"Count (N): {len(data)}\n"
                f"Mean:   {data.mean():.2f}\n"
                f"Median: {data.median():.2f}\n"
                f"StdDev: {data.std():.2f}\n"
                f"Min:    {data.min():.2f}\n"
                f"Max:    {data.max():.2f}"
            )
            
            # Place text box in upper right (outside plot if crowded, or inside)
            props = dict(boxstyle='round', facecolor='white', alpha=0.9)
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right', bbox=props)

        except Exception as e:
            logger.error(f"Plotting failed for {column}: {e}")
            plt.close(fig)
            return None
        
        # Aesthetics
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Save/Return
        if self.output_dir and save_name:
            try:
                out_path = self.output_dir / f"{save_name}.png"
                fig.savefig(out_path, dpi=150)
                if not return_fig: plt.close(fig)
            except Exception: pass

        if return_fig: return fig
        plt.show()
        return None

    def create_summary_page(self, comp_stats, exp_name):
        """Creates a text-based summary page figure."""
        if not comp_stats: return None
        
        fig = plt.figure(figsize=(8.5, 11)) # Letter size
        fig.clf()
        
        txt = f"REGISTRATION STABILITY SUMMARY: {exp_name}\n"
        txt += "="*60 + "\n\n"
        
        # Experiment Stats
        exp = comp_stats.get('experiment', {})
        txt += "GLOBAL STATISTICS\n"
        txt += f"- Total Nuclei Detected: {exp.get('total_detected', 0)}\n"
        txt += f"- Loss (FoV Method):     {exp.get('total_pruned_fov_method', 0)}\n"
        txt += f"- Loss (NE Method):      {exp.get('total_pruned_ne_method', 0)}\n"
        txt += f"- Net Difference:        {exp.get('net_loss_diff', 0)}\n"
        txt += "  (Positive = FoV Method is stricter)\n\n"
        
        # Parameters
        txt += "PARAMETERS\n"
        txt += f"- Pixel Size: {self.px_to_nm} nm\n"
        
        plt.text(0.1, 0.9, txt, transform=fig.transFigure, size=12, family='monospace', verticalalignment='top')
        plt.axis('off')
        return fig

    def generate_dashboard(self, reg_mode1_path, stability_ne_path, stability_fov_path, comparison_path=None, exp_name="Experiment", save_pdf=True):
        logger.info(f"Generating Dashboard for {exp_name}...")
        plt.close('all') 

        # 1. Load Data
        raw_reg = self.load_json(reg_mode1_path)
        ne_stab = self.load_json(stability_ne_path)
        fov_stab = self.load_json(stability_fov_path)
        comp_stats = self.load_json(comparison_path) if comparison_path else None
        
        # 2. Flatten
        df_raw = self.flatten_raw_registration(raw_reg)
        df_ne = self.flatten_ne_stability_report(ne_stab)
        df_fov = self.flatten_fov_stability_report(fov_stab)
        
        pdf = None
        if save_pdf and self.output_dir:
            try:
                pdf_path = self.output_dir / f"{exp_name}_Registration_Report.pdf"
                pdf = PdfPages(pdf_path) 
                logger.info(f"Opening PDF report: {pdf_path}")
            except Exception as e:
                logger.error(f"PDF creation failed: {e}")

        def save_plot(fig):
            if fig and pdf: pdf.savefig(fig); plt.close(fig)

        # --- PAGE 1: SUMMARY ---
        if comp_stats:
            fig = self.create_summary_page(comp_stats, exp_name)
            save_plot(fig)

        # --- PAGE 2+: RAW REGISTRATION ---
        if not df_raw.empty:
            fig = self.plot_histogram_with_fit(df_raw, 'angle_deg', f"{exp_name}: Rotation", "Angle (Degrees)", return_fig=True)
            save_plot(fig)
            fig = self.plot_histogram_with_fit(df_raw, 'scale_factor', f"{exp_name}: Scale", "Scale (Ratio)", return_fig=True)
            save_plot(fig)

        # --- PAGE 4+: STABILITY (NM) ---
        if not df_ne.empty:
            # Paper Figs 2c/d (Gaussian X/Y)
            fig = self.plot_histogram_with_fit(df_ne, 'drift_dx_nm', f"{exp_name}: NE Drift X", "dX (nm)", color='navy', return_fig=True)
            save_plot(fig)
            fig = self.plot_histogram_with_fit(df_ne, 'drift_dy_nm', f"{exp_name}: NE Drift Y", "dY (nm)", color='navy', return_fig=True)
            save_plot(fig)
            # Precision
            fig = self.plot_histogram_with_fit(df_ne, 'sigma_nm', f"{exp_name}: NE Precision (Noise Floor)", "Sigma (nm)", color='purple', return_fig=True)
            save_plot(fig)

        if not df_fov.empty:
            fig = self.plot_histogram_with_fit(df_fov, 'drift_dx_nm', f"{exp_name}: FoV Drift X", "dX (nm)", color='salmon', return_fig=True)
            save_plot(fig)
            fig = self.plot_histogram_with_fit(df_fov, 'drift_dy_nm', f"{exp_name}: FoV Drift Y", "dY (nm)", color='salmon', return_fig=True)
            save_plot(fig)

        if pdf: pdf.close()
        logger.info("Dashboard complete.")