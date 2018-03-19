

# importing all the required modules
import os
from collections import namedtuple
from test_textract import test_textract
from test_pydpf2 import test_pypdf2
from test_pyocr import test_pyocr

BL_NAME = "PARTNERSHIP COPY SIGN OBL.PDF"


def get_x_limits(line, line_header, column_header):
    """
    Returns x_min and x_max of the corresponding
    :param line_header:
    :param column_header:
    :return: a tuple x_min, x_max
    """
    if line.content != line_header:
        return None
    else:
        words_line = line_header.split(" ")
        words_column = column_header.split(" ")
        for i in range(len(words_line) - len(words_column) + 1):
            start = i
            end = i + len(words_column) - 1
            if words_column == words_line[start:end+1]:
                if start == 0:
                    x_min = 0
                else:
                    # x_min is the initial x of the first word of the column_header (left-aligned)
                    x_min = line.word_boxes[start].position[0][0]
                if end >= len(words_line)-1:
                    x_max = 99999
                else:
                    # x_max is the initial x of the first word AFTER the column_header (left-aligned)
                    x_max = line.word_boxes[end+1].position[0][0]
                return x_min, x_max


def parse_bl(ocr_text):
    parsed = dict()

    ParserConfig = namedtuple('ParserConfig', ['line_header', 'column_header', 'skip_lines'])

    parser_configs = [
            ParserConfig(line_header="Vessel Port of loading",
                      column_header="Vessel", skip_lines=0),
            ParserConfig(line_header="Vessel Port of loading",
                      column_header="Port of loading", skip_lines=0),
            ParserConfig(line_header="Port of discharge",
                      column_header="Port of discharge", skip_lines=0),
            ParserConfig(line_header="Shipper’s description of goods Gross weight",
                      column_header="Gross weight", skip_lines=0),
            ParserConfig(line_header="Freight payable at Place and date of issue",
                      column_header="Place and date of issue", skip_lines=1),
    ]

    for page in ocr_text:
        for i in range(len(page)):
            line = page[i]
            for config in parser_configs:
                if line.content == config.line_header:
                    x_min, x_max = get_x_limits(line, config.line_header, config.column_header)
                    words = []
                    skip_lines = config.skip_lines
                    for i_next in range(i+1, len(page)):
                        if page[i_next].content != "":
                            if skip_lines > 0:
                                skip_lines -= 1
                            else:
                                for word_box in page[i_next].word_boxes:
                                    if (x_min - 10) <= word_box.position[0][0] and \
                                        (x_max - 10) >= word_box.position[1][0]:
                                        words.append(word_box.content)
                                parsed[config.column_header] = " ".join(words)
                                break   # it just reads the first non-blank line
    return parsed


if __name__ == "__main__":
    filename = os.path.join('.', BL_NAME)
    #test_pypdf2(filename)
    #print(test_textract(filename))
    ocr_text = test_pyocr(filename)
    print(ocr_text)
    parsed_text = parse_bl(ocr_text)
    print(parsed_text)