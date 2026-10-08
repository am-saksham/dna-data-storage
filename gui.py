import sys
import os
import time
import math
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, 
                             QGraphicsLineItem, QFrame, QTabWidget, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

# Ensure we can import the compiled rust module
sys.path.append(os.path.join(os.path.dirname(__file__), "rust_engine", "target", "wheels")) 
import dna_codec

# Try to import Deep Learning dependencies
try:
    import torch
    sys.path.append(os.path.join(os.path.dirname(__file__), "ai_pipeline"))
    from model import DNADenoiserTransformer, DNAVocabulary
    HAS_AI = True
except ImportError:
    HAS_AI = False


class AIDecoderThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, fasta_dna):
        super().__init__()
        self.fasta_dna = fasta_dna
        
    def run(self):
        try:
            if not HAS_AI:
                self.error.emit("PyTorch AI libraries not found. Cannot run neural network.")
                return
                
            weights_path = os.path.join(os.path.dirname(__file__), "ai_pipeline", "weights", "dnadenoiser_sota.pth")
            if not os.path.exists(weights_path):
                self.log.emit("WARNING: No trained AI Brain found in weights folder. Skipping AI Denoising and jumping straight to Rust Decoder...")
                time.sleep(2)
                self.finished.emit(self.fasta_dna)
                return
                
            self.log.emit("BOOTING UP PYTORCH NEURAL NETWORK...")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            vocab = DNAVocabulary()
            model = DNADenoiserTransformer(vocab_size=vocab.vocab_size).to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
            model.eval()
            
            self.log.emit("NEURAL NETWORK ACTIVE. DENOISING SEQUENCE...")
            
            # Since the model was trained on ~400 nucleotide chunks, we mathematically chunk the file for inference
            chunk_size = 400 
            chunks = [self.fasta_dna[i:i+chunk_size] for i in range(0, len(self.fasta_dna), chunk_size)]
            
            clean_dna_pieces = []
            with torch.no_grad():
                for i, chunk in enumerate(chunks):
                    # Encode
                    src_tensor = torch.tensor([vocab.encode(chunk)], dtype=torch.long).to(device)
                    # For Seq2Seq inference without auto-regression (fast mode), we just push it through
                    tgt_dummy = src_tensor.clone() 
                    logits = model(src_tensor, tgt_dummy)
                    predicted_indices = torch.argmax(logits, dim=-1)[0]
                    clean_piece = vocab.decode(predicted_indices)
                    clean_dna_pieces.append(clean_piece)
                    
                    # Update progress bar
                    prog = int(((i + 1) / len(chunks)) * 100)
                    self.progress.emit(prog)
                    
            clean_full_dna = "".join(clean_dna_pieces)
            
            # Post-process: remove SOS, EOS, and PAD tokens physically
            clean_full_dna = clean_full_dna.replace("A", "A").replace("C", "C").replace("G", "G").replace("T", "T")
            valid_bases = [c for c in clean_full_dna if c in "ACGT"]
            
            self.log.emit("DENOISING COMPLETE. HANDING TO RUST ENGINE...")
            self.finished.emit("".join(valid_bases))
            
        except Exception as e:
            self.error.emit(f"AI ERROR: {str(e)}")


class NucleotideItem(QGraphicsEllipseItem):
    """Custom interactive graphic item for a single nucleotide"""
    def __init__(self, x, y, r, base, comp, index, app_ref, is_front):
        super().__init__(x - r, y - r, r * 2, r * 2)
        self.base = base
        self.comp = comp
        self.index = index
        self.app_ref = app_ref
        self.is_front = is_front
        
        self.setAcceptHoverEvents(True)
        colors = {'A': '#39FF14', 'C': '#00FFFF', 'G': '#FFEA00', 'T': '#FF073A'}
        self.setBrush(QBrush(QColor(colors.get(base, '#FFFFFF'))))
        
        if is_front:
            self.setPen(QPen(QColor("#FFFFFF"), 1))
            self.setZValue(2)
        else:
            self.setPen(QPen(QColor("#111111"), 1))
            self.setZValue(0)
            
    def hoverEnterEvent(self, event):
        self.app_ref.update_side_panel(self.index, self.base, self.comp)
        pen = self.pen()
        pen.setWidth(3)
        pen.setColor(QColor("#FFFFFF"))
        self.setPen(pen)
        super().hoverEnterEvent(event)
        
    def hoverLeaveEvent(self, event):
        pen = self.pen()
        pen.setWidth(1)
        pen.setColor(QColor("#FFFFFF") if self.is_front else QColor("#111111"))
        self.setPen(pen)
        super().hoverLeaveEvent(event)


class DNAApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DNA Storage System - Pro Edition")
        self.resize(1400, 900)
        
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
            QFrame#sidePanel { background-color: #16161A; border-left: 1px solid #2A2A35; }
            QFrame#metricsPanel { background-color: #16161A; border: 1px solid #2A2A35; border-radius: 8px; }
            QTabWidget::pane { border: none; background-color: #0F0F13; }
            QTabBar::tab {
                background: #16161A;
                color: #888;
                border: 1px solid #2A2A35;
                padding: 12px 40px;
                font-size: 16px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #007ACC;
                color: white;
                border: 1px solid #0099FF;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)
        
        self.tabs = QTabWidget()
        self.tab_encoder = QWidget()
        self.tab_decoder = QWidget()
        
        self.tabs.addTab(self.tab_encoder, "ENCODE FILE (BYTES → DNA)")
        self.tabs.addTab(self.tab_decoder, "DECODE & AI RECOVER (DNA → BYTES)")
        
        self.setup_encoder_tab()
        self.setup_decoder_tab()
        
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
        
        instruction = QLabel("Hover over any node in the matrix\\nto inspect the exact physical\\nmolecule constraints.")
        instruction.setStyleSheet("font-size: 14px; color: #444444; margin-top: 30px;")
        instruction.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        side_layout.addWidget(title)
        side_layout.addWidget(self.lbl_info_idx)
        side_layout.addWidget(self.lbl_info_base)
        side_layout.addWidget(self.lbl_info_comp)
        side_layout.addWidget(self.lbl_info_class)
        side_layout.addWidget(self.lbl_info_bonds)
        side_layout.addWidget(self.lbl_info_mass)
        side_layout.addWidget(instruction)
        side_layout.addStretch()
        
        main_layout.addWidget(self.tabs, stretch=1)
        main_layout.addWidget(self.side_panel)

    def setup_encoder_tab(self):
        layout = QVBoxLayout(self.tab_encoder)
        layout.setContentsMargins(20,20,20,20)
        
        controls = QHBoxLayout()
        btn_browse = QPushButton("ENCODE FILE")
        btn_browse.clicked.connect(self.encode_file)
        
        self.btn_save = QPushButton("EXPORT DNA (.FASTA)")
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
        
        metrics_frame = QFrame()
        metrics_frame.setObjectName("metricsPanel")
        metrics_layout = QHBoxLayout(metrics_frame)
        self.lbl_metrics = QLabel("No Data Loaded")
        metrics_layout.addWidget(self.lbl_metrics)
        
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setStyleSheet("background-color: #050505; border: 1px solid #2A2A35; border-radius: 8px;")
        
        layout.addLayout(controls)
        layout.addWidget(metrics_frame)
        layout.addWidget(self.view, stretch=1)

    def setup_decoder_tab(self):
        layout = QVBoxLayout(self.tab_decoder)
        layout.setContentsMargins(20,20,20,20)
        
        controls = QHBoxLayout()
        btn_browse = QPushButton("UPLOAD NOISY DNA (.FASTA)")
        btn_browse.clicked.connect(self.start_ai_decode)
        btn_browse.setStyleSheet("background-color: #FF073A; color: white;")
        
        self.lbl_decode_status = QLabel("WAITING FOR SEQUENCE")
        self.lbl_decode_status.setStyleSheet("color: #FFEA00; font-weight: bold; font-size: 16px;")
        
        controls.addWidget(btn_browse)
        controls.addSpacing(20)
        controls.addWidget(self.lbl_decode_status)
        controls.addStretch()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #2A2A35; border-radius: 5px; text-align: center; background: #16161A; color: white; font-weight: bold; }
            QProgressBar::chunk { background-color: #007ACC; width: 10px; }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        
        self.lbl_decode_metrics = QLabel("Ready to denoise.")
        self.lbl_decode_metrics.setStyleSheet("color: #888; font-size: 14px; margin-top: 10px; margin-bottom: 20px;")
        
        self.decode_scene = QGraphicsScene()
        self.decode_view = QGraphicsView(self.decode_scene)
        self.decode_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.decode_view.setStyleSheet("background-color: #050505; border: 1px solid #2A2A35; border-radius: 8px;")
        
        layout.addLayout(controls)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.lbl_decode_metrics)
        layout.addWidget(self.decode_view, stretch=1)
        
        self.recovered_bytes = None

    def start_ai_decode(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Noisy DNA File", "", "FASTA Files (*.fasta);;Text Files (*.txt)")
        if not filepath: return
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            fasta_dna = "".join(l.strip() for l in lines if not l.startswith(">"))
            
        self.lbl_decode_status.setText("INITIALIZING DEEP LEARNING MODEL...")
        self.lbl_decode_metrics.setText(f"Loaded {len(fasta_dna):,} noisy nucleotides.")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Start AI in background thread to prevent freezing GUI
        self.ai_thread = AIDecoderThread(fasta_dna)
        self.ai_thread.progress.connect(self.progress_bar.setValue)
        self.ai_thread.log.connect(self.lbl_decode_status.setText)
        self.ai_thread.finished.connect(self.run_rust_decoder)
        self.ai_thread.error.connect(self.show_ai_error)
        self.ai_thread.start()

    def run_rust_decoder(self, clean_dna):
        self.lbl_decode_status.setText("RUST ENGINE: REVERSING REED-SOLOMON MATH & ZSTD DECOMPRESSION...")
        QApplication.processEvents()
        
        try:
            start = time.time()
            recovered_bytes = dna_codec.decode_full_pipeline(clean_dna)
            elapsed = time.time() - start
            
            self.recovered_bytes = recovered_bytes
            
            self.lbl_decode_status.setText(f"FILE RECOVERED SUCCESSFULLY ({elapsed:.3f}s)!")
            self.lbl_decode_status.setStyleSheet("color: #39FF14; font-weight: bold; font-size: 18px;")
            self.lbl_decode_metrics.setText(f"Perfectly reconstructed original file: {len(recovered_bytes):,} bytes.")
            self.progress_bar.hide()
            
            # Prompt user to save the recovered file
            savepath, _ = QFileDialog.getSaveFileName(self, "Save Recovered File", "recovered_file.bin")
            if savepath:
                with open(savepath, 'wb') as f:
                    f.write(recovered_bytes)
                self.lbl_decode_metrics.setText(f"File saved perfectly to: {savepath}")
                
            self.draw_dna(clean_dna, target_scene=self.decode_scene)
            
        except Exception as e:
            self.lbl_decode_status.setText("DECODE FAILED: DNA TOO CORRUPTED")
            self.lbl_decode_status.setStyleSheet("color: #FF073A; font-weight: bold; font-size: 16px;")
            self.lbl_decode_metrics.setText(f"Rust Error: {str(e)}")
            self.progress_bar.hide()

    def show_ai_error(self, err):
        self.lbl_decode_status.setText("AI DENOISER CRASHED")
        self.lbl_decode_status.setStyleSheet("color: #FF073A; font-weight: bold; font-size: 16px;")
        self.lbl_decode_metrics.setText(str(err))
        self.progress_bar.hide()

    def update_side_panel(self, idx, base, comp):
        self.lbl_info_idx.setText(f"Position: {idx:,}")
        color_map = {'A': '#39FF14', 'C': '#00FFFF', 'G': '#FFEA00', 'T': '#FF073A'}
        color = color_map.get(base, '#FFF')
        
        self.lbl_info_base.setText(f'Base: <span style="color:{color}; font-weight:900; font-size:48px;">{base}</span>')
        self.lbl_info_comp.setText(f"Complement: {comp}")
        
        struct = "Purine (Double Ring)" if base in ['A', 'G'] else "Pyrimidine (Single Ring)"
        bonds = 3 if base in ['G', 'C'] else 2
        masses = {'A': 313.2, 'C': 289.2, 'G': 329.2, 'T': 304.2}
        
        self.lbl_info_class.setText(f"Structure: {struct}")
        self.lbl_info_bonds.setText(f"H-Bonds: {bonds}")
        self.lbl_info_mass.setText(f"Molar Mass: {masses.get(base, 0)} g/mol")

    def draw_dna(self, dna_string, target_scene=None):
        scene = target_scene if target_scene else self.scene
        scene.clear()
        
        def get_comp(b): return {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}.get(b, 'N')
        
        height, columns, y_offset, col_width, row_height = 40, 5, 30, 160, 20
        chunk_size = height * columns
        
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
                    
                    line = QGraphicsLineItem(x1, y, x2, y)
                    line.setPen(QPen(QColor("#333333"), 2))
                    line.setZValue(1)
                    scene.addItem(line)
                    
                    r = 6
                    if math.cos(t) > 0:
                        n1 = NucleotideItem(x1, y, r, base, comp, idx, self, True)
                        n2 = NucleotideItem(x2, y, r, comp, base, idx, self, False)
                    else:
                        n1 = NucleotideItem(x1, y, r, base, comp, idx, self, False)
                        n2 = NucleotideItem(x2, y, r, comp, base, idx, self, True)
                        
                    scene.addItem(n1)
                    scene.addItem(n2)
                    
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
                
            start = time.time()
            dna_string = dna_codec.encode_full_pipeline(file_bytes)
            elapsed = time.time() - start
            
            gc = dna_codec.get_gc_content(dna_string) * 100
            homo = dna_codec.get_max_homopolymer(dna_string)
            
            self.lbl_metrics.setText(f"<span style='color:#888;'>Size:</span> <span style='color:#FFF;'>{len(file_bytes):,} bytes</span> &nbsp;|&nbsp; <span style='color:#888;'>DNA Length:</span> <span style='color:#FFF;'>{len(dna_string):,} nt</span> &nbsp;|&nbsp; <span style='color:#888;'>GC Content:</span> <span style='color:#FFF;'>{gc:.2f}%</span>")
            self.lbl_status.setText(f"SUCCESS ({elapsed:.3f}s)")
            self.lbl_status.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 16px;")
            
            self.current_dna_string = dna_string
            self.btn_save.setEnabled(True)
            self.btn_save.setStyleSheet("background-color: #007ACC; color: white; border: 1px solid #0099FF;")
            
            self.draw_dna(dna_string)
            
        except Exception as e:
            self.lbl_status.setText("FATAL ERROR")
            self.lbl_status.setStyleSheet("color: #FF073A; font-weight: bold; font-size: 16px;")

    def save_dna(self):
        if not self.current_dna_string: return
        filepath, _ = QFileDialog.getSaveFileName(self, "Export DNA Sequence", "encoded_data.fasta", "FASTA Files (*.fasta);;Text Files (*.txt)")
        if filepath:
            with open(filepath, 'w') as f:
                if filepath.endswith('.fasta'):
                    f.write(">Synthetic_DNA_Storage_Payload_v1\\n")
                    for i in range(0, len(self.current_dna_string), 80):
                        f.write(self.current_dna_string[i:i+80] + "\\n")
                else:
                    f.write(self.current_dna_string)
            self.lbl_status.setText(f"EXPORTED TO {os.path.basename(filepath)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DNAApp()
    window.show()
    sys.exit(app.exec())
