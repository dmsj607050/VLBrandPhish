from __future__ import annotations

import argparse
import copy
import os
import shutil
import tkinter as tk
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple

from PIL import Image, ImageTk

IMG_ROOT = os.environ.get("VLBP_IMAGE_ROOT", "")
XML_ROOT = os.environ.get("VLBP_XML_ROOT", "")
CLASS_NAME = "logo"
KEEP_TOP1 = True
PROCESSED_FILE = ".processed_samples.txt"
SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
MIN_BOX = 4
HANDLE = 6


@dataclass
class Box:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: str = ""
    pose: str = "Unspecified"
    truncated: str = "0"
    difficult: str = "0"

    def normalize(self) -> None:
        self.xmin, self.xmax = sorted((int(self.xmin), int(self.xmax)))
        self.ymin, self.ymax = sorted((int(self.ymin), int(self.ymax)))

    def clamp(self, w: int, h: int) -> None:
        self.normalize()
        self.xmin = max(0, min(w - 1, self.xmin))
        self.xmax = max(0, min(w - 1, self.xmax))
        self.ymin = max(0, min(h - 1, self.ymin))
        self.ymax = max(0, min(h - 1, self.ymax))


@dataclass
class Record:
    xml_path: Path
    img_path: Path
    width: int
    height: int
    depth: int
    boxes: List[Box] = field(default_factory=list)


def indent(elem: ET.Element, level: int = 0) -> None:
    prefix = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = prefix + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = prefix
    elif level and (not elem.tail or not elem.tail.strip()):
        elem.tail = prefix


def image_size(path: Path) -> Tuple[int, int, int]:
    with Image.open(path) as im:
        return im.size[0], im.size[1], len(im.getbands())


def load_record(xml_path: Path, img_path: Path, class_name: str) -> Record:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        w, h, d = image_size(img_path)
    else:
        w = int(size.findtext("width", "0") or 0)
        h = int(size.findtext("height", "0") or 0)
        d = int(size.findtext("depth", "3") or 3)
    boxes: List[Box] = []
    for obj in root.findall("object"):
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        try:
            b = Box(
                name=obj.findtext("name") or class_name,
                xmin=int(float(bnd.findtext("xmin", "0") or 0)),
                ymin=int(float(bnd.findtext("ymin", "0") or 0)),
                xmax=int(float(bnd.findtext("xmax", "0") or 0)),
                ymax=int(float(bnd.findtext("ymax", "0") or 0)),
                confidence=obj.findtext("confidence", "") or "",
                pose=obj.findtext("pose", "Unspecified") or "Unspecified",
                truncated=obj.findtext("truncated", "0") or "0",
                difficult=obj.findtext("difficult", "0") or "0",
            )
            b.normalize()
            boxes.append(b)
        except ValueError:
            continue
    return Record(xml_path=xml_path, img_path=img_path, width=w, height=h, depth=d, boxes=boxes)


def save_record(rec: Record) -> None:
    ann = ET.Element("annotation")
    ET.SubElement(ann, "folder").text = rec.img_path.parent.name
    ET.SubElement(ann, "filename").text = rec.img_path.name
    ET.SubElement(ann, "path").text = str(rec.img_path)
    src = ET.SubElement(ann, "source")
    ET.SubElement(src, "database").text = "Unknown"
    size = ET.SubElement(ann, "size")
    ET.SubElement(size, "width").text = str(rec.width)
    ET.SubElement(size, "height").text = str(rec.height)
    ET.SubElement(size, "depth").text = str(rec.depth)
    ET.SubElement(ann, "segmented").text = "0"
    for b in rec.boxes:
        b.clamp(rec.width, rec.height)
        obj = ET.SubElement(ann, "object")
        ET.SubElement(obj, "name").text = b.name
        ET.SubElement(obj, "pose").text = b.pose
        ET.SubElement(obj, "truncated").text = b.truncated
        ET.SubElement(obj, "difficult").text = b.difficult
        if b.confidence != "":
            ET.SubElement(obj, "confidence").text = b.confidence
        bb = ET.SubElement(obj, "bndbox")
        ET.SubElement(bb, "xmin").text = str(b.xmin)
        ET.SubElement(bb, "ymin").text = str(b.ymin)
        ET.SubElement(bb, "xmax").text = str(b.xmax)
        ET.SubElement(bb, "ymax").text = str(b.ymax)
    indent(ann)
    ET.ElementTree(ann).write(rec.xml_path, encoding="utf-8", xml_declaration=True)


def keep_top_box(rec: Record) -> None:
    if len(rec.boxes) <= 1:
        return

    def score(box: Box) -> float:
        try:
            return float(box.confidence)
        except (TypeError, ValueError):
            return -1.0

    rec.boxes = [max(rec.boxes, key=score)]


def scan_dataset(xml_root: Path, img_root: Path) -> List[Tuple[Path, Path]]:
    out: List[Tuple[Path, Path]] = []
    for xml in sorted(xml_root.rglob("*.xml")):
        rel = xml.relative_to(xml_root)
        folder = img_root / rel.parent
        for suf in SUFFIXES:
            img = folder / (xml.stem + suf)
            if img.is_file():
                out.append((xml, img))
                break
    return out


class Editor(tk.Tk):
    def __init__(self, img_root: Path, xml_root: Path, class_name: str, keep_top1: bool) -> None:
        super().__init__()
        self.title("Logo标注可视化调整工具")
        self.geometry("1650x980")
        self.img_root = img_root
        self.xml_root = xml_root
        self.class_name = class_name
        self.keep_top1 = keep_top1
        self.processed_path = self.xml_root / PROCESSED_FILE
        self.items = scan_dataset(xml_root, img_root)
        if not self.items:
            raise SystemExit("没有找到可用的 XML 和图片配对。")

        self.processed = self.load_processed()
        self.filter_var = tk.StringVar()
        self.process_filter_var = tk.StringVar(value="全部")
        self.last_filter_state: Optional[Tuple[str, str]] = None
        self.visible: List[int] = []

        self.idx = 0
        self.rec: Optional[Record] = None
        self.img: Optional[Image.Image] = None
        self.imgtk: Optional[ImageTk.PhotoImage] = None
        self.cache_size: Optional[Tuple[int, int]] = None
        self.scale = 1.0
        self.sel: Optional[int] = None
        self.drag_mode: Optional[str] = None
        self.drag_start: Optional[Tuple[int, int]] = None
        self.anchor: Optional[Box] = None

        self.build()
        self.bind_all("<KeyPress>", self.on_key)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.load(0)

    def load_processed(self) -> set[str]:
        if not self.processed_path.is_file():
            return set()
        return {line.strip() for line in self.processed_path.read_text(encoding="utf-8").splitlines() if line.strip()}

    def save_processed(self) -> None:
        text = "\n".join(sorted(self.processed))
        self.processed_path.write_text(text + ("\n" if text else ""), encoding="utf-8")

    def relkey(self, xml_path: Path) -> str:
        return str(xml_path.relative_to(self.xml_root)).replace("\\", "/")

    def mark_processed(self) -> None:
        if self.rec is None:
            return
        key = self.relkey(self.rec.xml_path)
        if key not in self.processed:
            self.processed.add(key)
            self.save_processed()
            self.refresh_files(force=True)

    def build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        bar = ttk.Frame(self, padding=6)
        bar.grid(row=0, column=0, columnspan=3, sticky="ew")
        bar.columnconfigure(11, weight=1)
        ttk.Button(bar, text="上一张(A)", command=self.prev_item).grid(row=0, column=0, padx=4)
        ttk.Button(bar, text="下一张(D)", command=self.next_item).grid(row=0, column=1, padx=4)
        ttk.Button(bar, text="保存", command=self.save_current).grid(row=0, column=2, padx=4)
        ttk.Button(bar, text="放大", command=lambda: self.zoom(1.2)).grid(row=0, column=3, padx=4)
        ttk.Button(bar, text="缩小", command=lambda: self.zoom(1 / 1.2)).grid(row=0, column=4, padx=4)
        ttk.Button(bar, text="适应窗口", command=self.fit).grid(row=0, column=5, padx=4)
        ttk.Button(bar, text="批量仅保留最高框", command=self.batch_keep_top1).grid(row=0, column=6, padx=4)
        ttk.Button(bar, text="删除当前样本", command=self.delete_sample).grid(row=0, column=7, padx=4)
        ttk.Label(bar, text="处理状态").grid(row=0, column=8, padx=(12, 4))
        combo = ttk.Combobox(
            bar,
            textvariable=self.process_filter_var,
            values=("全部", "只看已处理", "只看未处理"),
            state="readonly",
            width=12,
        )
        combo.grid(row=0, column=9, padx=(0, 8))
        combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_files(force=True))
        ttk.Label(bar, text="名称筛选").grid(row=0, column=10, padx=(12, 4))
        ent = ttk.Entry(bar, textvariable=self.filter_var, width=30)
        ent.grid(row=0, column=11, sticky="ew")
        ent.bind("<KeyRelease>", lambda _e: self.refresh_files(force=True))

        self.file_list = tk.Listbox(self, exportselection=False, width=28)
        self.file_list.grid(row=1, column=0, sticky="nsw", padx=(8, 4), pady=8)
        self.file_list.bind("<<ListboxSelect>>", self.on_file_pick)

        center = ttk.Frame(self)
        center.grid(row=1, column=1, sticky="nsew", padx=4, pady=8)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(center, bg="#202124", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        xs = ttk.Scrollbar(center, orient="horizontal", command=self.canvas.xview)
        ys = ttk.Scrollbar(center, orient="vertical", command=self.canvas.yview)
        xs.grid(row=1, column=0, sticky="ew")
        ys.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=xs.set, yscrollcommand=ys.set)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Double-Button-1>", self.on_double_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<MouseWheel>", self.on_wheel)

        right = ttk.Frame(self, padding=(4, 8, 8, 8))
        right.grid(row=1, column=2, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="标注框").grid(row=0, column=0, sticky="w")
        self.box_list = tk.Listbox(right, exportselection=False, width=24)
        self.box_list.grid(row=1, column=0, sticky="nsew")
        self.box_list.bind("<<ListboxSelect>>", self.on_box_pick)
        btns = ttk.Frame(right)
        btns.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(btns, text="删除选中框", command=self.delete_box).grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="类别设为logo", command=self.rename_box).grid(row=0, column=1, padx=4)

        self.status = tk.StringVar(value="已就绪")
        nav_zone = tk.Label(
            right,
            text="快速切图区\n左键：上一张\n右键：下一张",
            justify="center",
            bg="#2b2b2b",
            fg="#f2f2f2",
            padx=10,
            pady=12,
            relief="ridge",
            bd=2,
            cursor="hand2",
        )
        nav_zone.grid(row=3, column=0, sticky="sew", pady=(12, 0))
        nav_zone.bind("<Button-1>", lambda _e: self.prev_item())
        nav_zone.bind("<Button-3>", lambda _e: self.next_item())

        ttk.Label(self, textvariable=self.status, anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        self.refresh_files(force=True)

    def sync_file_selection(self) -> None:
        if not self.visible or not self.items:
            return
        current = self.items[self.idx][0]
        for list_idx, data_idx in enumerate(self.visible):
            if self.items[data_idx][0] == current:
                self.file_list.selection_clear(0, tk.END)
                self.file_list.selection_set(list_idx)
                self.file_list.see(list_idx)
                break

    def refresh_files(self, force: bool = False) -> None:
        state = (self.filter_var.get().strip().lower(), self.process_filter_var.get())
        if not force and state == self.last_filter_state:
            self.sync_file_selection()
            return

        query, proc_filter = state
        self.file_list.delete(0, tk.END)
        self.visible = []
        for i, (xml, _img) in enumerate(self.items):
            rel = self.relkey(xml)
            done = rel in self.processed
            if query and query not in rel.lower():
                continue
            if proc_filter == "只看已处理" and not done:
                continue
            if proc_filter == "只看未处理" and done:
                continue
            self.visible.append(i)
            prefix = "[已处理] " if done else "[未处理] "
            self.file_list.insert(tk.END, prefix + rel)
        self.last_filter_state = state
        self.sync_file_selection()

    def load(self, idx: int) -> None:
        self.idx = max(0, min(idx, len(self.items) - 1))
        xml, img = self.items[self.idx]
        self.rec = load_record(xml, img, self.class_name)
        rel = self.relkey(xml)
        if self.keep_top1 and rel not in self.processed:
            keep_top_box(self.rec)
            save_record(self.rec)
        self.img = Image.open(img).convert("RGB")
        self.imgtk = None
        self.cache_size = None
        self.scale = 1.0
        self.sel = 0 if self.rec.boxes else None
        self.mark_processed()
        self.refresh_files()
        self.refresh_boxes()
        self.fit()
        self.update_status()
        self.canvas.focus_set()

    def save_current(self) -> None:
        if self.rec is None:
            return
        save_record(self.rec)
        self.mark_processed()
        self.update_status()

    def update_status(self) -> None:
        if self.rec is None:
            return
        processed_count = len(self.processed)
        self.status.set(
            "第 {}/{} 张 | 已处理 {}/{} | {} | {} 个框 | 缩放 {:.2f}x{}".format(
                self.idx + 1,
                len(self.items),
                processed_count,
                len(self.items),
                self.relkey(self.rec.xml_path),
                len(self.rec.boxes),
                self.scale,
                " | 仅保留最高置信度框" if self.keep_top1 else "",
            )
        )

    def fit(self) -> None:
        if self.img is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        iw, ih = self.img.size
        self.scale = max(0.05, min(cw / iw, ch / ih))
        self.imgtk = None
        self.cache_size = None
        self.redraw()
        self.update_status()

    def zoom(self, factor: float) -> None:
        self.scale = max(0.05, min(8.0, self.scale * factor))
        self.imgtk = None
        self.cache_size = None
        self.redraw()
        self.update_status()

    def redraw(self) -> None:
        self.canvas.delete("all")
        if self.img is None or self.rec is None:
            return
        iw, ih = self.img.size
        sw = max(1, int(round(iw * self.scale)))
        sh = max(1, int(round(ih * self.scale)))
        if self.imgtk is None or self.cache_size != (sw, sh):
            disp = self.img if (sw, sh) == (iw, ih) else self.img.resize((sw, sh), Image.Resampling.BILINEAR)
            self.imgtk = ImageTk.PhotoImage(disp)
            self.cache_size = (sw, sh)
        self.canvas.create_image(0, 0, anchor="nw", image=self.imgtk)
        self.canvas.configure(scrollregion=(0, 0, sw, sh))
        for i, b in enumerate(self.rec.boxes):
            x1, y1 = self.to_canvas(b.xmin, b.ymin)
            x2, y2 = self.to_canvas(b.xmax, b.ymax)
            color = "#00ff66" if i == self.sel else "#ff5252"
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
            self.canvas.create_text(x1 + 4, max(12, y1 - 8), anchor="sw", fill=color, text=f"{i} {b.name}")
            if i == self.sel:
                for hx, hy in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.canvas.create_rectangle(hx - HANDLE, hy - HANDLE, hx + HANDLE, hy + HANDLE, fill=color, outline=color)

    def to_canvas(self, x: int, y: int) -> Tuple[int, int]:
        return int(round(x * self.scale)), int(round(y * self.scale))

    def to_image(self, ex: int, ey: int) -> Tuple[int, int]:
        if self.rec is None:
            return 0, 0
        x = int(round(self.canvas.canvasx(ex) / self.scale))
        y = int(round(self.canvas.canvasy(ey) / self.scale))
        return max(0, min(self.rec.width - 1, x)), max(0, min(self.rec.height - 1, y))

    def refresh_boxes(self) -> None:
        self.box_list.delete(0, tk.END)
        if self.rec is None:
            return
        for i, b in enumerate(self.rec.boxes):
            extra = f" conf={b.confidence}" if b.confidence else ""
            self.box_list.insert(tk.END, f"#{i:02d} {b.name} [{b.xmin}, {b.ymin}, {b.xmax}, {b.ymax}]{extra}")
        self.box_list.selection_clear(0, tk.END)
        if self.sel is not None and self.rec.boxes:
            self.box_list.selection_set(self.sel)
            self.box_list.see(self.sel)

    def on_file_pick(self, _event: tk.Event) -> None:
        if not self.file_list.curselection():
            return
        dataset_idx = self.visible[self.file_list.curselection()[0]]
        if dataset_idx != self.idx:
            self.load(dataset_idx)

    def on_box_pick(self, _event: tk.Event) -> None:
        self.sel = self.box_list.curselection()[0] if self.box_list.curselection() else None
        self.redraw()
        self.canvas.focus_set()

    def hit_box(self, x: int, y: int) -> Optional[int]:
        if self.rec is None:
            return None
        for i in range(len(self.rec.boxes) - 1, -1, -1):
            b = self.rec.boxes[i]
            if b.xmin <= x <= b.xmax and b.ymin <= y <= b.ymax:
                return i
        return None

    def hit_handle(self, ex: int, ey: int) -> Optional[str]:
        if self.rec is None or self.sel is None:
            return None
        b = self.rec.boxes[self.sel]
        x1, y1 = self.to_canvas(b.xmin, b.ymin)
        x2, y2 = self.to_canvas(b.xmax, b.ymax)
        cx, cy = self.canvas.canvasx(ex), self.canvas.canvasy(ey)
        for name, (hx, hy) in {"tl": (x1, y1), "tr": (x2, y1), "bl": (x1, y2), "br": (x2, y2)}.items():
            if abs(cx - hx) <= HANDLE * 2 and abs(cy - hy) <= HANDLE * 2:
                return name
        return None

    def persist(self) -> None:
        self.save_current()
        self.refresh_boxes()
        self.redraw()

    def on_press(self, event: tk.Event) -> None:
        if self.rec is None:
            return
        self.canvas.focus_set()
        x, y = self.to_image(event.x, event.y)
        self.drag_start = (x, y)
        self.anchor = None
        handle = self.hit_handle(event.x, event.y)
        if handle is not None and self.sel is not None:
            self.drag_mode = handle
            self.anchor = copy.deepcopy(self.rec.boxes[self.sel])
            return
        hit = self.hit_box(x, y)
        if hit is not None:
            self.sel = hit
            self.drag_mode = "move"
            self.anchor = copy.deepcopy(self.rec.boxes[hit])
            self.refresh_boxes()
            self.redraw()
            return
        self.rec.boxes.append(Box(name=self.class_name, xmin=x, ymin=y, xmax=x, ymax=y))
        self.sel = len(self.rec.boxes) - 1
        self.drag_mode = "create"
        self.anchor = copy.deepcopy(self.rec.boxes[self.sel])
        self.refresh_boxes()
        self.redraw()

    def on_drag(self, event: tk.Event) -> None:
        if self.rec is None or self.sel is None or self.drag_mode is None or self.drag_start is None or self.anchor is None:
            return
        x, y = self.to_image(event.x, event.y)
        sx, sy = self.drag_start
        a = self.anchor
        b = self.rec.boxes[self.sel]
        if self.drag_mode == "move":
            dx, dy = x - sx, y - sy
            b.xmin, b.xmax = a.xmin + dx, a.xmax + dx
            b.ymin, b.ymax = a.ymin + dy, a.ymax + dy
        elif self.drag_mode == "create":
            b.xmin, b.ymin, b.xmax, b.ymax = sx, sy, x, y
        elif self.drag_mode == "tl":
            b.xmin, b.ymin, b.xmax, b.ymax = x, y, a.xmax, a.ymax
        elif self.drag_mode == "tr":
            b.xmin, b.ymin, b.xmax, b.ymax = a.xmin, y, x, a.ymax
        elif self.drag_mode == "bl":
            b.xmin, b.ymin, b.xmax, b.ymax = x, a.ymin, a.xmax, y
        elif self.drag_mode == "br":
            b.xmin, b.ymin, b.xmax, b.ymax = a.xmin, a.ymin, x, y
        b.normalize()
        b.clamp(self.rec.width, self.rec.height)
        self.refresh_boxes()
        self.redraw()

    def on_release(self, _event: tk.Event) -> None:
        if self.rec is None:
            self.drag_mode = None
            self.anchor = None
            return
        if self.sel is not None and self.sel < len(self.rec.boxes):
            b = self.rec.boxes[self.sel]
            if (b.xmax - b.xmin) < MIN_BOX or (b.ymax - b.ymin) < MIN_BOX:
                self.rec.boxes.pop(self.sel)
                self.sel = None if not self.rec.boxes else min(self.sel, len(self.rec.boxes) - 1)
        self.drag_mode = None
        self.anchor = None
        self.persist()

    def on_double_click(self, event: tk.Event) -> None:
        if self.rec is None:
            return
        x, y = self.to_image(event.x, event.y)
        hit = self.hit_box(x, y)
        if hit is None:
            return
        self.sel = hit
        self.refresh_boxes()
        self.redraw()
        if messagebox.askyesno("删除提醒", "是否删除当前选中的框？"):
            self.delete_box()

    def on_right_click(self, event: tk.Event) -> None:
        if self.rec is None:
            return
        x, y = self.to_image(event.x, event.y)
        hit = self.hit_box(x, y)
        if hit is None:
            return
        self.sel = hit
        self.delete_box()

    def on_wheel(self, event: tk.Event) -> None:
        self.zoom(1.1 if event.delta > 0 else 1 / 1.1)

    def move_box(self, dx: int, dy: int) -> None:
        if self.rec is None or self.sel is None:
            return
        b = self.rec.boxes[self.sel]
        b.xmin += dx
        b.xmax += dx
        b.ymin += dy
        b.ymax += dy
        b.clamp(self.rec.width, self.rec.height)
        self.persist()

    def delete_box(self) -> None:
        if self.rec is None or self.sel is None:
            return
        self.rec.boxes.pop(self.sel)
        self.sel = None if not self.rec.boxes else min(self.sel, len(self.rec.boxes) - 1)
        self.persist()

    def rename_box(self) -> None:
        if self.rec is None or self.sel is None:
            return
        self.rec.boxes[self.sel].name = self.class_name
        self.persist()

    def prev_item(self) -> None:
        if self.idx > 0:
            self.load(self.idx - 1)

    def next_item(self) -> None:
        if self.idx < len(self.items) - 1:
            self.load(self.idx + 1)

    def batch_keep_top1(self) -> None:
        if not messagebox.askyesno("批量清理确认", "将只保留每张图置信度最高的 1 个框，并直接覆盖 XML。是否继续？"):
            return
        total = 0
        changed = 0
        for xml, img in self.items:
            rec = load_record(xml, img, self.class_name)
            before = len(rec.boxes)
            keep_top_box(rec)
            save_record(rec)
            total += 1
            if len(rec.boxes) < before:
                changed += 1
        self.load(self.idx)
        messagebox.showinfo("批量清理完成", f"处理完成。\n总文件数：{total}\n发生删框的文件数：{changed}")

    def delete_sample(self) -> None:
        if self.rec is None:
            return
        img_dir = self.rec.img_path.parent
        xml_dir = self.rec.xml_path.parent
        if not messagebox.askyesno("删除当前样本", f"将删除：\n图片文件夹：{img_dir}\n标注文件夹：{xml_dir}\n\n该操作不可撤销，是否继续？"):
            return
        rel_dir = str(self.rec.xml_path.relative_to(self.xml_root).parent).replace("\\", "/")
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if xml_dir.exists():
            shutil.rmtree(xml_dir)
        self.processed = {k for k in self.processed if not (k == rel_dir or k.startswith(rel_dir + "/"))}
        self.save_processed()
        self.items = scan_dataset(self.xml_root, self.img_root)
        self.last_filter_state = None
        if not self.items:
            self.file_list.delete(0, tk.END)
            self.box_list.delete(0, tk.END)
            self.canvas.delete("all")
            self.status.set("所有样本都已删除。")
            return
        self.refresh_files(force=True)
        self.load(min(self.idx, len(self.items) - 1))

    def on_key(self, event: tk.Event) -> Optional[str]:
        key = str(getattr(event, "keysym", "") or "")
        state = int(getattr(event, "state", 0) or 0)
        shift = bool(state & 0x0001)
        if key in ("a", "A"):
            self.prev_item()
            return "break"
        if key in ("d", "D"):
            self.next_item()
            return "break"
        if key == "Left":
            self.move_box(-10 if shift else -1, 0)
            return "break"
        if key == "Right":
            self.move_box(10 if shift else 1, 0)
            return "break"
        if key == "Up":
            self.move_box(0, -10 if shift else -1)
            return "break"
        if key == "Down":
            self.move_box(0, 10 if shift else 1)
            return "break"
        if key == "Delete":
            self.delete_box()
            return "break"
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="可视化编辑 Pascal VOC XML logo 标注。")
    parser.add_argument("--images-root", type=str, default=IMG_ROOT)
    parser.add_argument("--xml-root", type=str, default=XML_ROOT)
    parser.add_argument("--class-name", type=str, default=CLASS_NAME)
    parser.add_argument("--keep-top1", action="store_true", default=KEEP_TOP1)
    parser.add_argument("--keep-all", dest="keep_top1", action="store_false")
    args = parser.parse_args()

    img_root = Path(os.path.expanduser(args.images_root)).resolve()
    xml_root = Path(os.path.expanduser(args.xml_root)).resolve()
    if not img_root.is_dir():
        raise SystemExit(f"图片目录不存在：{img_root}")
    if not xml_root.is_dir():
        raise SystemExit(f"XML 标注目录不存在：{xml_root}")

    app = Editor(img_root, xml_root, args.class_name, bool(args.keep_top1))
    app.mainloop()


if __name__ == "__main__":
    main()
