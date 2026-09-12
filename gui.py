import sys
import os
import time
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, 
                             QGraphicsLineItem, QFrame)
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QPainter, QFont

# Ensure we can import the compiled rust module
sys.path.append(os.path.join(os.path.dirname(__file__), "dna_codec", "target", "wheels")) 
import dna_codec

class NucleotideItem(QGraphicsEllipseItem):
    """Custom interactive graphic item for a single nucleotide"""
    def __init__(self, x, y, r, base, comp, index, app_ref, is_front):
        # Center the ellipse at (x,y)
        super().__init__(x - r, y - r, r * 2, r * 2)
        self.base = base
        self.comp = comp
        self.index = index
        self.app_ref = app_ref
        self.is_front = is_front
        
        # Enable hover events
        self.setAcceptHoverEvents(True)
        
        # Professional High-Tech Colors
        colors = {'A': '#39FF14', 'C': '#00FFFF', 'G': '#FFEA00', 'T': '#FF073A'}
        
        self.setBrush(QBrush(QColor(colors.get(base, '#FFFFFF'))))
        
        if is_front:
            self.setPen(QPen(QColor("#FFFFFF"), 1))
            self.setZValue(2)
        else:
            self.setPen(QPen(QColor("#111111"), 1))
            self.setZValue(0)
            
    def hoverEnterEvent(self, event):
        # Update the side panel on hover
        self.app_ref.update_side_panel(self.index, self.base, self.comp)
        
        # Highlight effect: widen the border
        pen = self.pen()
        pen.setWidth(3)
        pen.setColor(QColor("#FFFFFF"))
        self.setPen(pen)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        # Remove highlight
        pen = self.pen()
        pen.setWidth(1)
        pen.setColor(QColor("#FFFFFF") if self.is_front else QColor("#111111"))
        self.setPen(pen)
        super().hoverLeaveEvent(event)


class DNAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNA Codec Engine - Professional Edition")
        self.resize(1400, 900)
        
        # Modern Dark Theme CSS
        self.setStyleSheet("""
            QMainWindow { background-color: #0F0F13; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI', Arial; font-size: 14px; }
            QPushButton { 
                background-color: #2D2D36; 
                color: white; 
                border: 1px solid #4A4A5A; 
                padding: 10px 20px; 
                border-radius: 6px; 
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3D3D4A; border: 1px solid #5A5A6A; }
            QPushButton:pressed { background-color: #1D1D26; }
            QFrame#sidePanel { 
                background-color: #16161A; 
                border-left: 1px solid #2A2A35; 
            }
            QFrame#metricsPanel {
                background-color: #16161A;
                border: 1px solid #2A2A35;
                border-radius: 8px;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)
        
        # --- LEFT SIDE (Controls & Matrix) ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(20,20,20,20)
        left_layout.setSpacing(15)
        
        # Controls
        controls = QHBoxLayout()
        btn_browse = QPushButton("ENCODE FILE")
        btn_browse.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse.clicked.connect(self.encode_file)
        
        self.btn_save = QPushButton("EXPORT DNA (.FASTA)")
        self.btn_save.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save.clicked.connect(self.save_dna)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("background-color: #005080; color: #888;")
        
        self.lbl_status = QLabel("SYSTEM IDLE")
        self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 16px;")
        
        controls.addWidget(btn_browse)
        controls.addWidget(self.btn_save)
        controls.addSpacing(20)
        controls.addWidget(self.lbl_status)
        controls.addStretch()
        
        self.current_dna_string = None
        
        # Metrics Panel
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metricsPanel")
        metrics_layout = QHBoxLayout(metrics_frame)
        self.lbl_metrics = QLabel("No Data Loaded")
        metrics_layout.addWidget(self.lbl_metrics)
        
        # DNA Graphics Canvas
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        # Hardware anti-aliasing removes the "cartoonish" pixelated edges
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("""
            QGraphicsView {
                background-color: #050505; 
                border: 1px solid #2A2A35;
                border-radius: 8px;
            }
        """)
        
        left_layout.addLayout(controls)
        left_layout.addWidget(metrics_frame)
        left_layout.addWidget(self.view, stretch=1)
        
        # --- RIGHT SIDE (Hover Info Panel) ---
        self.side_panel = QFrame()
        self.side_panel.setObjectName("sidePanel")
        self.side_panel.setFixedWidth(350)
        side_layout = QVBoxLayout(self.side_panel)
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        side_layout.setContentsMargins(30, 40, 30, 40)
        
        title = QLabel("MOLECULE INSPECTOR")
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #5A5A6A; letter-spacing: 2px;")
        
        self.lbl_info_idx = QLabel("Position: -")
        self.lbl_info_idx.setStyleSheet("font-size: 20px; color: #FFFFFF; margin-top: 20px;")
        
        self.lbl_info_base = QLabel("Base: -")
        self.lbl_info_base.setStyleSheet("font-size: 24px; margin-top: 10px;")
        
        self.lbl_info_comp = QLabel("Complement: -")
        self.lbl_info_comp.setStyleSheet("font-size: 16px; color: #888888; margin-top: 5px;")
        
        self.lbl_info_class = QLabel("Structure: -")
        self.lbl_info_class.setStyleSheet("font-size: 14px; color: #777777; margin-top: 15px;")
        
        self.lbl_info_bonds = QLabel("H-Bonds: -")
        self.lbl_info_bonds.setStyleSheet("font-size: 14px; color: #777777; margin-top: 5px;")
        
        self.lbl_info_mass = QLabel("Molar Mass: -")
        self.lbl_info_mass.setStyleSheet("font-size: 14px; color: #777777; margin-top: 5px;")
        
        self.lbl_info_chunk = QLabel("Reed-Solomon Chunk: -")
        self.lbl_info_chunk.setStyleSheet("font-size: 14px; color: #666666; margin-top: 25px;")
        
        instruction = QLabel("Hover over any node in the matrix\\nto inspect the exact physical\\nmolecule constraints.")
        instruction.setStyleSheet("font-size: 14px; color: #444444; margin-top: 30px;")
        instruction.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        glossary_title = QLabel("SCIENTIFIC GLOSSARY")
        glossary_title.setStyleSheet("font-size: 14px; font-weight: 800; color: #5A5A6A; margin-top: 30px; letter-spacing: 1px;")
        
        glossary_text = QLabel(
            "<b>GC Content:</b> Ratio of G and C bases. Must stay near 50% so the DNA doesn't physically melt or fold on itself.<br><br>"
            "<b>Homopolymer:</b> Repeating sequences (e.g. AAAA). They cause lab synthesizers to crash. Our engine mathematically prevents them.<br><br>"
            "<b>Complement:</b> The opposite strand of the double-helix. Adenine (A) always pairs with Thymine (T). Cytosine (C) pairs with Guanine (G).<br><br>"
            "<b>Reed-Solomon Chunk:</b> Data is split into mathematical blocks. If physical DNA mutates or degrades, the RS equations perfectly rebuild the missing file."
        )
        glossary_text.setWordWrap(True)
        glossary_text.setStyleSheet("font-size: 13px; color: #777777; line-height: 1.5;")
        
        side_layout.addWidget(title)
        side_layout.addWidget(self.lbl_info_idx)
        side_layout.addWidget(self.lbl_info_base)
        side_layout.addWidget(self.lbl_info_comp)
        side_layout.addWidget(self.lbl_info_class)
        side_layout.addWidget(self.lbl_info_bonds)
        side_layout.addWidget(self.lbl_info_mass)
        side_layout.addWidget(self.lbl_info_chunk)
        side_layout.addWidget(instruction)
        side_layout.addWidget(glossary_title)
        side_layout.addWidget(glossary_text)
        side_layout.addStretch()
        
        main_layout.addWidget(left_widget, stretch=1)
        main_layout.addWidget(self.side_panel)

    def update_side_panel(self, idx, base, comp):
        self.lbl_info_idx.setText(f"Position: {idx:,}")
        
        color_map = {'A': '#39FF14', 'C': '#00FFFF', 'G': '#FFEA00', 'T': '#FF073A'}
        color = color_map.get(base, '#FFF')
        
        self.lbl_info_base.setText(f'Base: <span style="color:{color}; font-weight:900; font-size:48px;">{base}</span>')
        self.lbl_info_comp.setText(f"Complement: {comp}")
        
        if base in ['A', 'G']:
            struct = "Purine (Double Ring)"
        else:
            struct = "Pyrimidine (Single Ring)"
            
        bonds = 3 if base in ['G', 'C'] else 2
        masses = {'A': 313.2, 'C': 289.2, 'G': 329.2, 'T': 304.2}
        
        self.lbl_info_class.setText(f"Structure: {struct}")
        self.lbl_info_bonds.setText(f"H-Bonds: {bonds}")
        self.lbl_info_mass.setText(f"Molar Mass: {masses.get(base, 0)} g/mol")
        
        # Calculate which 255-byte chunk this nucleotide belongs to
        chunk_num = idx // (255 * 3) # Approx chunk size map
        self.lbl_info_chunk.setText(f"Block Matrix ID: {chunk_num}")

    def draw_dna(self, dna_string):
        self.scene.clear()
        
        def get_comp(b): return {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}.get(b, 'N')
        
        height = 40
        columns = 5
        chunk_size = height * columns
        y_offset = 30
        col_width = 160
        row_height = 20
        
        # Render a large chunk for high-tech visualization
        display_limit = 4000
        display_dna = dna_string[:display_limit]
        
        for block_start in range(0, len(display_dna), chunk_size):
            block_dna = display_dna[block_start:block_start+chunk_size]
            
            for c in range(columns):
                col_dna = block_dna[c*height : (c+1)*height]
                base_x = 80 + c * col_width
                
                for row, base in enumerate(col_dna):
                    idx = block_start + c*height + row
                    comp = get_comp(base)
                    
                    t = row * 0.35
                    amplitude = 40
                    
                    x1 = base_x + amplitude * math.sin(t)
                    x2 = base_x + amplitude * math.sin(t + math.pi)
                    y = y_offset + row * row_height
                    
                    # Bridge
                    line = QGraphicsLineItem(x1, y, x2, y)
                    line.setPen(QPen(QColor("#333333"), 2))
                    line.setZValue(1)
                    self.scene.addItem(line)
                    
                    r = 6
                    if math.cos(t) > 0:
                        n1 = NucleotideItem(x1, y, r, base, comp, idx, self, True)
                        n2 = NucleotideItem(x2, y, r, comp, base, idx, self, False)
                    else:
                        n1 = NucleotideItem(x1, y, r, base, comp, idx, self, False)
                        n2 = NucleotideItem(x2, y, r, comp, base, idx, self, True)
                        
                    self.scene.addItem(n1)
                    self.scene.addItem(n2)
                    
            y_offset += height * row_height + 50

    def encode_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select File to Encode to DNA")
        if not filepath: return
        
        self.lbl_status.setText("ENCODING DATA STREAM...")
        self.lbl_status.setStyleSheet("color: #FFEA00; font-weight: bold; font-size: 16px;")
        QApplication.processEvents()
        
        try:
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
                
            start_time = time.time()
            dna_string = dna_codec.encode_full_pipeline(file_bytes)
            elapsed = time.time() - start_time
            
            gc = dna_codec.get_gc_content(dna_string) * 100
            homo = dna_codec.get_max_homopolymer(dna_string)
            
            # Update Metrics Panel
            metrics_html = f"""
            <span style="color:#888;">Original File Size:</span> <span style="color:#FFF;">{len(file_bytes):,} bytes</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#888;">Encoded DNA Length:</span> <span style="color:#FFF;">{len(dna_string):,} nucleotides</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#888;">GC Content:</span> <span style="color:#FFF;">{gc:.2f}%</span> &nbsp;&nbsp;|&nbsp;&nbsp;
            <span style="color:#888;">Max Homopolymer:</span> <span style="color:#FFF;">{homo}</span>
            """
            self.lbl_metrics.setText(metrics_html)
            
            self.lbl_status.setText(f"SUCCESS ({elapsed:.3f}s)")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 16px;")
            
            self.current_dna_string = dna_string
            self.btn_save.setEnabled(True)
            self.btn_save.setStyleSheet("background-color: #007ACC; color: white; border: 1px solid #0099FF;")
            
            self.draw_dna(dna_string)
            
        except Exception as e:
            self.lbl_status.setText("FATAL ERROR")
            self.lbl_status.setStyleSheet("color: #FF073A; font-weight: bold; font-size: 16px;")
            print(f"Error: {e}")

    def save_dna(self):
        if not self.current_dna_string:
            return
        filepath, _ = QFileDialog.getSaveFileName(self, "Export DNA Sequence", "encoded_data.fasta", "FASTA Files (*.fasta);;Text Files (*.txt)")
        if filepath:
            try:
                with open(filepath, 'w') as f:
                    if filepath.endswith('.fasta'):
                        f.write(">Synthetic_DNA_Storage_Payload_v1\\n")
                        # FASTA standard: 80 characters per line
                        for i in range(0, len(self.current_dna_string), 80):
                            f.write(self.current_dna_string[i:i+80] + "\\n")
                    else:
                        f.write(self.current_dna_string)
                self.lbl_status.setText(f"EXPORTED TO {os.path.basename(filepath)}")
                self.lbl_status.setStyleSheet("color: #00FFFF; font-weight: bold; font-size: 16px;")
            except Exception as e:
                self.lbl_status.setText("EXPORT ERROR")
                self.lbl_status.setStyleSheet("color: #FF073A; font-weight: bold; font-size: 16px;")
                print(f"Save Error: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNAApp()
    window.show()
    sys.exit(app.exec())
