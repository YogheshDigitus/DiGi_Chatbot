import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import fitz
from pathlib import Path
from itertools import islice
from utils.categorizer import PDFTextBlockCategorizer


class PDFExtractor:
    pdf_root = "..."

    def __init__(self):
        pdf_filename = "***.pdf"
        self.pdf_fullpath = self.pdf_root / pdf_filename
        self.pdf_doc = fitz.open(self.pdf_fullpath)

    def calc_rect_center(self, rect, reverse_y=False):
        if reverse_y:
            x0, y0, x1, y1 = rect[0], -rect[1], rect[2], -rect[3]
        else:
            x0, y0, x1, y1 = rect

        x_center = (x0 + x1) / 2
        y_center = (y0 + y1) / 2
        return (x_center, y_center)

    def extract_all_text_blocks(self):
        # * https://pymupdf.readthedocs.io/en/latest/textpage.html#TextPage.extractBLOCKS

        rect_centers = []
        rects = []
        visual_label_texts = []
        categorize_vectors = []

        for page_idx, page in islice(enumerate(self.pdf_doc), len(self.pdf_doc)):
            blocks = page.get_text("blocks")
            page_cnt = page_idx + 1
            print(f"=== Start Page {page_cnt}: {len(blocks)} blocks ===")
            block_cnt = 0
            for block in blocks:
                block_rect = block[:4]  # (x0,y0,x1,y1)
                x0, y0, x1, y1 = block_rect
                rects.append(block_rect)
                block_text = block[4]
                block_num = block[5]
                # block_cnt += 1
                block_cnt = block_num + 1

                rect_center = self.calc_rect_center(block_rect, reverse_y=True)
                rect_centers.append(rect_center)
                # visual_label_text = f"{block_text.split()[-1]}({page_cnt}.{block_cnt})"
                visual_label_text = f"({page_cnt}.{block_cnt})"
                visual_label_texts.append(visual_label_text)

                block_type = "text" if block[6] == 0 else "image"
                print(f"Block: {page_cnt}.{block_cnt}")
                print(f"<{block_type}> {rect_center} - {block_rect}")
                print(block_text)
                categorize_vectors.append((*block_rect, block_text))

            print(f"=== End Page {page_cnt}: {len(blocks)} blocks ===\n")

        categorizer = PDFTextBlockCategorizer(categorize_vectors)
        categorizer.run()

        fig, ax = plt.subplots()
        colors = ["b", "r", "g", "c", "m", "y", "k"]

        for i, rect_center in enumerate(rect_centers):
            label_idx = categorizer.labels[i]
            color = colors[label_idx]
            x0, y0, x1, y1 = rects[i]
            rect = Rectangle((x0, -y0), x1 - x0, -y1 + y0, fill=False, edgecolor=color)
            ax.add_patch(rect)
            x, y = rect_center
            plt.scatter(x, y, color=color)
            plt.annotate(visual_label_texts[i], rect_center)
        plt.show()

    def run(self):
        self.extract_all_text_blocks()


if __name__ == "__main__":
    pdf_extractor = PDFExtractor()
    pdf_extractor.run()