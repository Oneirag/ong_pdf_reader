

# importing all the required modules
import os
from collections import namedtuple
from test_textract import test_textract
from test_pydpf2 import test_pypdf2
from test_pyocr import test_pyocr

BL_NAME = "pdfs/NBA Van Gogh - Original BLs.pdf"
BL_NAME = "pds/CAPE LILY V.25L OBL.PDF"
#BL_NAME = "pdfs/img-427173626.pdf"

def strcomp(str1, str2):
    str2 = ''.join(ch for ch in str2 if ch.isalnum() or ch==",")
    str1 = ''.join(ch for ch in str1 if ch.isalnum())
    if len(str2) > len(str1):
        return False
    if str1.upper()==str2.upper():
        return True
    same_chars = 0
    for char1, char2 in zip(str1, str2):
        if char1.upper()==char2.upper():
            same_chars += 1
    #if same_chars == len(str2): return True
    return (same_chars / len(str1)) > 0.7    # If more than 80% of chars are equals, return true

class ParserConfig(object):
    def __init__(self, line_header, column_header, skip_lines=0, function=None):
        if not isinstance(line_header, str):
            self.aliases_line_header = line_header
        else:
            self.aliases_line_header = [line_header ,]

        if not isinstance(column_header, str):
            self.aliases_column_header = column_header
        else:
            self.aliases_column_header = [column_header ,]
        self.skip_lines = skip_lines
        self.__used = False
        self.function = function

    def check_line(self, line):
        if self.__used:
            return False
        if line != "":
            for line_header, column_header in zip(self.aliases_line_header, self.aliases_column_header):
                if strcomp(line_header, line) and len(line) > line_header.find(column_header):
                    self.line_header = line_header
                    self.column_header = column_header
                    self.__used = True
                    return True
        return False

    def check_words(self, words):
        if self.function:
            return self.function(words)
        else:
            return words

    def used(self):
        return self.__used


def get_x_limits(line, line_header, column_header):
    """
    Returns x_min and x_max of the corresponding
    :param line_header:
    :param column_header:
    :return: a tuple x_min, x_max
    """
    if not strcomp(line_header, line.content):
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

                    # x_min is the initial x of the first word of the column_header (left-aligned)
                    x_min = 0.5 * (line.word_boxes[start].position[0][0] +
                                   line.word_boxes[start - 1].position[1][0])

                if end >= len(words_line)-1:
                    x_max = 99999
                else:
                    # x_max is the initial x of the first word AFTER the column_header (left-aligned)
                    x_max = line.word_boxes[end+1].position[0][0]

                    # x_max is the average of the final of the current word and the initial x of the first word AFTER the column_header (left-aligned)
                    x_max = 0.5 * (line.word_boxes[end + 1].position[0][0] +
                                   line.word_boxes[end].position[1][0])

                return x_min, x_max

def extract_date(words):
    if words[-1][-4:].isnumeric():
        if len(words[-1]) > 4:
            return [words[-1]]    # if format is DD-MM-YYYY
        else:
            return words[-3:]   # if format is MONTH DAY, YEAR
    else:
        return None

def parse_bl(ocr_text):
    parsed = dict()

    parser_configs = (
        ParserConfig(line_header=["Vessel Port of loading",
                                  "Name of Vessel Port of loading",],
                     column_header=["Vessel", "Name of Vessel"] , skip_lines=0),
        ParserConfig(line_header=["Vessel Port of loading",
                                  "Name of Vessel Port of loading",],
                     column_header=["Port of loading", "Port of loading"], skip_lines=0),
        ParserConfig(line_header=["Port of discharge", ],
                     column_header=["Port of discharge",], skip_lines=0),
        ParserConfig(line_header=["Shippers description of goods Gross weight",
                                  "Shippers description of goods Net weight",
                                  'Gross weight',],
                     column_header=["Gross weight", "Net weight", "Gross weight"],
                        skip_lines=0),
            ParserConfig(line_header=["Freight payable at Place and date of issue",
                                      "Place and date of issue"],
                      column_header=["Place and date of issue",
                                     "Place and date of issue"], skip_lines=1, function=extract_date),
    )

    for config in parser_configs:
        parsed[config.aliases_column_header[0]] = None


    bl = (line.content for line in ocr_text[0] if "BILL OF LADING" in line.content)
    if not bl:
        return None

    for page in ocr_text:
        all_configs_used = True
        for config in parser_configs:
            if not config.used():
                all_configs_used = False
                break
        if all_configs_used: break
        for i, line in enumerate(page):
            for config in parser_configs:
                if config.check_line(line.content):
                    x_min, x_max = get_x_limits(line, config.line_header, config.column_header)
                    skip_lines = config.skip_lines
                    for i_next in range(i+1, len(page)):
                        if page[i_next].content != "":
                            words = []
                            for word_box in page[i_next].word_boxes:
                                if (x_min - 20) <= word_box.position[0][0] and \
                                        (x_max + 20) >= word_box.position[1][0]:
                                    #                                    if (x_min) <= word_box.position[1][0] and \
                                    #                                            (x_max) >= word_box.position[0][0]:
                                    words.append(word_box.content)
                            sanitized_words = config.check_words(words)
                            if skip_lines > 0:
                                if sanitized_words:
                                    parsed[config.aliases_column_header[0]] = " ".join(sanitized_words)
                                    break  # it just reads the first non-blank line
                                skip_lines -= 1
                            else:
                                if sanitized_words:
                                    parsed[config.aliases_column_header[0]] = " ".join(words)
                                break   # it just reads the first non-blank line
    return parsed


if __name__ == "__main__":
    filename = os.path.join('.', BL_NAME)
    #test_pypdf2(filename)
    #print(test_textract(filename))
    ocr_text = test_pyocr(filename)
    for pageidx in range(len(ocr_text)):
        print("\n".join([ "[{}][{}] ".format(pageidx, idx) + line.content for idx, line in enumerate(ocr_text[pageidx])]))
    parsed_text = parse_bl(ocr_text)
    print(parsed_text)