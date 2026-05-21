from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import json
import os

class QAReport:
    def __init__(self, output_dir="reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_report(self, patient_id, results):
        filename = os.path.join(self.output_dir, f"QA_Report_{patient_id}.pdf")
        c = canvas.Canvas(filename, pagesize=letter)
        width, height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, f"Synthetic CT QA Report - Patient: {patient_id}")

        y_position = height - 80

        # Summary Section
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_position, "Summary Status:")
        y_position -= 20

        for module_name, module_res in results.items():
            status = module_res.get('status', 'UNKNOWN')
            color = colors.green if status == "PASS" else colors.red
            c.setFillColor(color)
            c.drawString(70, y_position, f"{module_name}: {status}")
            y_position -= 15

        c.setFillColor(colors.black)
        y_position -= 20
        c.line(50, y_position, width - 50, y_position)
        y_position -= 30

        # Detailed Metrics
        c.setFont("Helvetica", 10)
        for module_name, module_res in results.items():
            if y_position < 100:
                c.showPage()
                y_position = height - 50

            c.setFont("Helvetica-Bold", 11)
            c.drawString(50, y_position, f"Module: {module_name}")
            y_position -= 15
            c.setFont("Helvetica", 9)

            # Flatten dict for display
            flat_metrics = self._flatten_dict(module_res.get('results', {}))

            # Check for plots first
            plots = [v for k, v in module_res.get('results', {}).items() if k.startswith('plot_')]

            flat_metrics = self._flatten_dict(module_res.get('results', {}))

            for key, value in flat_metrics.items():
                if "plot_" in key: continue # Skip plot paths in text list

                if y_position < 50:
                    c.showPage()
                    y_position = height - 50
                    c.setFont("Helvetica", 9)

                # Format value
                if isinstance(value, float):
                    val_str = f"{value:.4f}"
                else:
                    val_str = str(value)

                c.drawString(70, y_position, f"{key}: {val_str}")
                y_position -= 12

            # Draw Plots
            for plot_path in plots:
                if os.path.exists(plot_path):
                    if y_position < 300: # Need space for image
                        c.showPage()
                        y_position = height - 50

                    # Draw image
                    # Keep aspect ratio roughly
                    c.drawImage(plot_path, 100, y_position - 250, width=400, height=250, preserveAspectRatio=True)
                    y_position -= 270

            y_position -= 15

        c.save()
        return filename

    def _flatten_dict(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Simple list handling
                items.append((new_key, f"List[{len(v)}]"))
            else:
                items.append((new_key, v))
        return dict(items)
