import sys
import os
import glob
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QListWidget, QListWidgetItem, QLabel, QProgressBar, 
    QRadioButton, QButtonGroup, QFrame, QMessageBox, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QGraphicsTextItem, QGraphicsItemGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QRectF
from PyQt6.QtGui import QPixmap, QColor, QFont, QPainter, QBrush, QPen, QPainterPath

from thermal_analysis import process_single_pair, get_temperature_at_pixel

# =========================================================================
# INTERACTIVE ZOOMABLE & CLICKABLE CANVAS
# =========================================================================
class ZoomableGraphicsView(QGraphicsView):
    pixel_clicked = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # Disable hand drag and force normal arrow cursor
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.ArrowCursor)

        # Set zoom to anchor directly under the mouse pointer
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self.zoom_factor = 1.15
        self.temp_markers = []
        self.is_clickable = False

    def wheelEvent(self, event):
        if self.pixmap_item.pixmap().isNull():
            return

        # Zoom in/out anchored at mouse position
        factor = self.zoom_factor if event.angleDelta().y() > 0 else (1 / self.zoom_factor)
        self.scale(factor, factor)
    def set_image(self, image_path, allow_click=False, save_view_transform=True):
        """Loads image into canvas without breaking current zoom/pan state if changing view modes."""
        self.is_clickable = allow_click
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            
            # Store transform to keep zoom position when switching tabs
            current_transform = self.transform() if save_view_transform else None
            
            self.pixmap_item.setPixmap(pixmap)
            self.scene.setSceneRect(0, 0, pixmap.width(), pixmap.height())

            if current_transform and not current_transform.isIdentity():
                self.setTransform(current_transform)
            else:
                self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
    def reset_zoom(self):
        self.resetTransform()
        if not self.pixmap_item.pixmap().isNull():
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self.pixmap_item.pixmap().isNull():
            return
        factor = self.zoom_factor if event.angleDelta().y() > 0 else (1 / self.zoom_factor)
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        # Left-click to inspect temperature ONLY when clickable mode is active
        if event.button() == Qt.MouseButton.LeftButton and self.is_clickable:
            if not self.pixmap_item.pixmap().isNull():
                scene_pos = self.mapToScene(event.pos())
                x, y = int(scene_pos.x()), int(scene_pos.y())
                pixmap = self.pixmap_item.pixmap()
                
                if 0 <= x < pixmap.width() and 0 <= y < pixmap.height():
                    self.pixel_clicked.emit(x, y)

        # Right-click to clear placed markers
        elif event.button() == Qt.MouseButton.RightButton:
            self.clear_markers()

        super().mousePressEvent(event)

    def add_temp_marker(self, x, y, temp_c):
        """Draws semi-transparent temperature card over clicked point."""
        marker_group = QGraphicsItemGroup()
        self.scene.addItem(marker_group)

        # Draw red/cyan target point
        r = 6
        dot = self.scene.addEllipse(
            x - r, y - r, r * 2, r * 2, 
            QPen(QColor("#00FFFF"), 2), 
            QBrush(QColor("#FF0055"))
        )
        marker_group.addToGroup(dot)

        # Temperature label string
        text_str = f"{temp_c:.1f}°C"
        text_item = QGraphicsTextItem(text_str)
        text_item.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        text_item.setDefaultTextColor(QColor("white"))

        text_rect = text_item.boundingRect()
        padding_x, padding_y = 8, 4
        card_w = text_rect.width() + (padding_x * 2)
        card_h = text_rect.height() + (padding_y * 2)
        
        offset_x = x + 12
        offset_y = y - (card_h / 2)

        # Semi-transparent background box
        card_rect = QRectF(offset_x, offset_y, card_w, card_h)
        path = QPainterPath()
        path.addRoundedRect(card_rect, 5, 5)

        bg_card = self.scene.addPath(
            path,
            QPen(QColor("#00FFFF"), 1.5),
            QBrush(QColor(15, 15, 20, 180))
        )
        
        text_item.setPos(offset_x + padding_x, offset_y + padding_y)
        
        marker_group.addToGroup(bg_card)
        marker_group.addToGroup(text_item)
        marker_group.setZValue(10)

        self.temp_markers.append(marker_group)

    def clear_markers(self):
        for item in self.temp_markers:
            self.scene.removeItem(item)
        self.temp_markers.clear()

# =========================================================================
# ASYNC WORKER THREAD
# =========================================================================
class AnalysisWorker(QThread):
    progress = pyqtSignal(int, int)
    item_analyzed = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)

    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path

    def run(self):
        output_dir = os.path.join(self.folder_path, "thermal_output_results")
        
        thermal_files = glob.glob(os.path.join(self.folder_path, "*_T.[jJ][pP][gG]")) + \
                        glob.glob(os.path.join(self.folder_path, "*_T.[jJ][pP][eE][gG]"))

        results = {}
        total = len(thermal_files)

        for idx, t_path in enumerate(thermal_files):
            base_prefix = t_path.rsplit('_T.', 1)[0]
            ext = os.path.splitext(t_path)[1]

            mask_path = f"{base_prefix}_T.png"
            wide_path = f"{base_prefix}_W{ext}"

            if not os.path.exists(mask_path):
                mask_path = f"{base_prefix}_T.PNG"

            filename = os.path.basename(t_path)

            if os.path.exists(mask_path):
                try:
                    res_t, res_rgb = process_single_pair(t_path, mask_path, wide_path, output_dir)
                    pair_res = {'thermal': res_t, 'rgb': res_rgb}
                    results[filename] = pair_res
                    self.item_analyzed.emit(filename, pair_res)
                except Exception as e:
                    print(f"Error processing {t_path}: {e}")

            self.progress.emit(idx + 1, total)

        self.finished.emit(results)

# =========================================================================
# MAIN GUI WINDOW
# =========================================================================
class ThermalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thermal Anomaly Inspector")
        self.resize(1200, 750)
        self.processed_results = {}
        self.current_folder = ""
        self.list_items_map = {}
        self.current_thermal_path = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Panel
        left_panel = QFrame()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)

        self.btn_select = QPushButton("📁 Choose Folder")
        self.btn_select.clicked.connect(self.select_folder)
        left_layout.addWidget(self.btn_select)

        self.lbl_folder = QLabel("No directory selected")
        self.lbl_folder.setWordWrap(True)
        left_layout.addWidget(self.lbl_folder)

        self.btn_run = QPushButton("▶ Run Analysis")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.start_processing)
        left_layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)

        left_layout.addWidget(QLabel("Image List:"))
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self.on_file_changed)
        left_layout.addWidget(self.file_list)

        layout.addWidget(left_panel)

        # Right Panel
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)

        # 3 View Modes Radio Header
        toggle_layout = QHBoxLayout()
        self.radio_thermal_res = QRadioButton("Analyzed Thermal View")
        self.radio_orig_thermal = QRadioButton("Original Thermal View (Clickable)")
        self.radio_rgb = QRadioButton("Wide RGB View")
        
        self.radio_orig_thermal.setChecked(True)

        self.btn_group = QButtonGroup()
        self.btn_group.addButton(self.radio_thermal_res)
        self.btn_group.addButton(self.radio_orig_thermal)
        self.btn_group.addButton(self.radio_rgb)

        self.radio_thermal_res.toggled.connect(self.update_canvas)
        self.radio_orig_thermal.toggled.connect(self.update_canvas)
        self.radio_rgb.toggled.connect(self.update_canvas)

        self.btn_reset_zoom = QPushButton("🔍 Reset Zoom")
        self.btn_reset_zoom.clicked.connect(lambda: self.canvas.reset_zoom())

        toggle_layout.addWidget(self.radio_thermal_res)
        toggle_layout.addWidget(self.radio_orig_thermal)
        toggle_layout.addWidget(self.radio_rgb)
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.btn_reset_zoom)
        right_layout.addLayout(toggle_layout)

        self.canvas = ZoomableGraphicsView()
        self.canvas.setStyleSheet("border: 1px solid #444; background: #111;")
        self.canvas.pixel_clicked.connect(self.on_canvas_clicked)
        
        right_layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(right_panel, stretch=1)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.current_folder = folder
            self.lbl_folder.setText(folder)
            self.btn_run.setEnabled(True)
            self.processed_results.clear()
            self.list_items_map.clear()
            self.file_list.clear()
            
            thermal_files = glob.glob(os.path.join(self.current_folder, "*_T.[jJ][pP][gG]")) + \
                            glob.glob(os.path.join(self.current_folder, "*_T.[jJ][pP][eE][gG]"))
            
            for f in thermal_files:
                filename = os.path.basename(f)
                item = QListWidgetItem(f"⚪ {filename}")
                item.setForeground(QColor("#A0A0A0"))
                item.setData(Qt.ItemDataRole.UserRole, filename)
                
                self.file_list.addItem(item)
                self.list_items_map[filename] = item

            # Auto-select the first image to display it immediately on the canvas
            if self.file_list.count() > 0:
                self.file_list.setCurrentRow(0)

    def start_processing(self):
        self.btn_run.setEnabled(False)
        self.btn_select.setEnabled(False)
        
        self.worker = AnalysisWorker(self.current_folder)
        self.worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(int((cur/tot)*100)))
        self.worker.item_analyzed.connect(self.on_item_analyzed)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_item_analyzed(self, filename, result_data):
        self.processed_results[filename] = result_data
        
        if filename in self.list_items_map:
            item = self.list_items_map[filename]
            item.setText(f"🟢 {filename}")
            item.setForeground(QColor("#00E676"))
            item.setFont(QFont("Arial", 9, QFont.Weight.Bold))

        current_item = self.file_list.currentItem()
        if current_item and current_item.data(Qt.ItemDataRole.UserRole) == filename:
            self.update_canvas()

    def on_finished(self, results):
        self.btn_run.setEnabled(True)
        self.btn_select.setEnabled(True)
        QMessageBox.information(self, "Complete", "Batch processing completed successfully!")

    def on_file_changed(self):
        self.canvas.clear_markers()
        self.update_canvas()
        self.canvas.reset_zoom()

    def on_canvas_clicked(self, x, y):
        """Reads temperature from raw R-JPEG even if user clicked on Analyzed Thermal View."""
        # Enable temp reading for both original and analyzed thermal views
        if (self.radio_orig_thermal.isChecked() or self.radio_thermal_res.isChecked()) and self.current_thermal_path:
            temp = get_temperature_at_pixel(self.current_thermal_path, x, y)
            if temp is not None:
                self.canvas.add_temp_marker(x, y, temp)

    def update_canvas(self):
        item = self.file_list.currentItem()
        if not item:
            return

        filename = item.data(Qt.ItemDataRole.UserRole)
        self.current_thermal_path = os.path.join(self.current_folder, filename)

        base_prefix = filename.rsplit('_T.', 1)[0]
        ext = os.path.splitext(filename)[1]

        # Determine target path & click permissions
        if self.radio_orig_thermal.isChecked():
            img_path = self.current_thermal_path
            allow_click = True
        elif self.radio_thermal_res.isChecked():
            if filename in self.processed_results:
                img_path = self.processed_results[filename]['thermal']
            else:
                img_path = self.current_thermal_path
            allow_click = True
        else:
            if filename in self.processed_results:
                img_path = self.processed_results[filename]['rgb']
            else:
                img_path = os.path.join(self.current_folder, f"{base_prefix}_W{ext}")
            allow_click = False

        if img_path and os.path.exists(img_path):
            # Reset existing transformations and zoom levels completely
            self.canvas.resetTransform()
            
            # Pass save_view_transform=False so it scales to fit fresh every switch
            self.canvas.set_image(img_path, allow_click=allow_click, save_view_transform=False)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ThermalApp()
    win.show()
    sys.exit(app.exec())