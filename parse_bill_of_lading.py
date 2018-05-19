# importing all the required modules
import os
from collections import namedtuple
from poc_pyocr import test_pyocr
from collections import Counter



def init_parser_config():
    return (
        ParserConfig(column_header=["Vessel", "Name of Vessel"], skip_lines=0),
        ParserConfig(column_header=["Port of loading"], skip_lines=0),
        ParserConfig(column_header=["Port of discharge", ], skip_lines=0),
        ParserConfig(column_header=["Gross weight", "Net weight"], skip_lines=0),
        ParserConfig(column_header=["Place and date of issue",], skip_lines=2, function=extract_date),
    )


def my_strcomp(str1, str2):
    """
    Compares two strings, ignoring spaces, capitals, non numeric chars and allow 30% of different chars
    :param str1:
    :param str2:
    :return:
    """
    str2 = ''.join(ch for ch in str2 if ch.isalnum() or ch == ",")
    str1 = ''.join(ch for ch in str1 if ch.isalnum())
    if len(str2) > len(str1):
        return False
    if str1.upper() == str2.upper():
        return True
    same_chars = 0
    for char1, char2 in zip(str1, str2):
        if char1.upper() == char2.upper():
            same_chars += 1
    # if same_chars == len(str2): return True
    return (same_chars / len(str1)) > 0.7  # If more than 80% of chars are equals, return true



class ColumnText(object):

    def __init__(self, word_box):
        self.position = []
        for point in word_box.position:
            self.position.append([])
            for coord in point:
                self.position[-1].append(coord)
        self.content = word_box.content
        self.word_boxes = []
        self.word_boxes.append(word_box)

    def can_merge(self, word_box):
        return word_box.position[0][0] <= self.position[1][0] + \
                                          (self.position[1][1] - self.position[0][1]) * 1.5

    def merge(self, word_box):
        self.content += " " + word_box.content
        self.word_boxes.append(word_box)
        self.position[1][0] = word_box.position[1][0]
        self.position[1][1] = max(word_box.position[1][1], self.position[1][1])
        self.position[0][1] = min(word_box.position[0][1], self.position[0][1])

def get_paragraphs(page):
    lines = []
    for line in page:
        if line.content != "":
            new_line = []
            new_box = None
            for word in line.word_boxes:
                if word.content !="":
                    if not new_box:
                        new_box = ColumnText(word)
                        new_line.append(new_box)
                    else:
                        if new_box.can_merge(word):
                            new_box.merge(word)
                        else:
                            new_box = ColumnText(word)
                            new_line.append(new_box)
            lines.append(new_line)
    return lines

class ParserConfig(object):
    def __init__(self, line_header=[], column_header=[], skip_lines=0, function=None):
        if not isinstance(line_header, str):
            self.aliases_line_header = line_header
        else:
            self.aliases_line_header = [line_header, ]

        if not isinstance(column_header, str):
            self.aliases_column_header = column_header
        else:
            self.aliases_column_header = [column_header, ]
        self.skip_lines = skip_lines
        self.__used = False
        self.function = function

    def check_line(self, line):
        if self.__used:
            return False
        if line != "":
            #for line_header, column_header in zip(self.aliases_line_header, self.aliases_column_header):
            for column_header in self.aliases_column_header:
                #if line_header.startswith("Place and date ofissue"):
                #    pass
                #if my_strcomp(line_header, line):
                if my_strcomp(column_header, line):
                   # if len(line) > line_header.find(column_header):
                        self.line_header = column_header  # line_header
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
    if not my_strcomp(line_header, line.content):
        return None
    else:
        words_line = line_header.split(" ")
        words_column = column_header.split(" ")
        for i in range(len(words_line) - len(words_column) + 1):
            start = i
            end = i + len(words_column) - 1
            if words_column == words_line[start:end + 1]:
                if start == 0:
                    x_min = 0
                else:
                    # x_min is the initial x of the first word of the column_header (left-aligned)
                    x_min = line.word_boxes[start].position[0][0]

                    # x_min is the initial x of the first word of the column_header (left-aligned)
                    x_min = 0.5 * (line.word_boxes[start].position[0][0] +
                                   line.word_boxes[start - 1].position[1][0])

                if end >= len(words_line) - 1:
                    x_max = 99999
                else:
                    # x_max is the initial x of the first word AFTER the column_header (left-aligned)
                    x_max = line.word_boxes[end + 1].position[0][0]

                    # x_max is the average of the final of the current word and the initial x of the first word AFTER the column_header (left-aligned)
                    x_max = 0.5 * (line.word_boxes[end + 1].position[0][0] +
                                   line.word_boxes[end].position[1][0])

                return x_min, x_max


def extract_date(words):
    if len(words) == 1:
        words = words[0].split(" ")
    if words:
        if words[-1][-4:].isnumeric():
            if len(words[-1]) > 4:
                return [words[-1]]  # if format is DD-MM-YYYY
            else:
                return words[-3:]  # if format is MONTH DAY, YEAR
    return None


def sanitize_chars(item):
    """
    Removes non printable elements from char. If it is a number, only one "," is allowed, rest are changed by "."
    :param item:
    :return:
    """
    new_item = []
    for char in item:
        if char.isalnum() or char in " ,./-":
            new_item.append(char)
    if new_item[0].isnumeric():
        # look for duplicated ",", and replace the last for a dot if no dots are found
        if new_item.count(",") > 1 and new_item.count(".") == 0:
            new_item[len(new_item) - 1 - new_item[::-1].index(",")] = "."
    return "".join(new_item)



def parse_bl_with_counter(ocr_text):
    """
    Parses bl text
    :param ocr_text: must be a generator
    :return:
    """
    parsed = dict()

    parser_configs = init_parser_config()

    for config in parser_configs:
        parsed[config.aliases_column_header[0]] = None
        parsed[config.aliases_column_header[0]] = []


    for page in ocr_text:

        parser_configs = init_parser_config()
        lines = get_paragraphs(page)

        bl = (line.content for line in page if "BILL OF LADING" in line.content)
        if not bl:
            return None
        all_configs_used = True
        for config in parser_configs:
            if not config.used():
                all_configs_used = False
                break
        if all_configs_used: break
        for i, line in enumerate(lines):
            for word_block in line:
                for config in parser_configs:
                    if config.check_line(word_block.content):
                        x_min = word_block.position[0][0]
                        x_max = word_block.position[1][0]
                        #x_min, x_max = get_x_limits(line, config.line_header, config.column_header)
                        skip_lines = config.skip_lines
                        for i_next in range(i + 1, len(lines)):
                            #if page[i_next].content != "":
                                words = []
                                for block in lines[i_next]:  #.word_boxes:
                                    if block.position[0][0] <= x_max and \
                                        block.position[1][0] >= x_min:
                                        words.append(block.content)
                                sanitized_words = config.check_words(words)
                                if skip_lines > 0:
                                    if sanitized_words:
                                        parsed[config.aliases_column_header[0]].append(sanitize_chars(" ".join(sanitized_words)))
                                        break  # it just reads the first non-blank line
                                    skip_lines -= 1
                                else:
                                    if sanitized_words:
                                        parsed[config.aliases_column_header[0]].append(
                                            sanitize_chars(" ".join(words)))
                                    break  # it just reads the first non-blank line
    # For each concept, find the "winner" one, the one which at least has two equals. If none found, then the first one is used
    new_parsed = dict()
    counter_parsed = dict()

    for key, values in parsed.items():
        if values:
            if key == "Place and date of issue":
                print(values)
            c = Counter(values)
            new_value = c.most_common(1)[0][0]
            new_parsed[key] = new_value
            counter_parsed[key] = c.most_common(1)[0][1]
        else:
            new_parsed[key] = None
    return new_parsed, counter_parsed


def parse_bl(ocr_text):
    parsed, _ = parse_bl_with_counter(ocr_text)
    return parsed




if __name__ == "__main__":
    from config_endesa import config_bill_of_lading
    BL_NAME = config_bill_of_lading.keys()[-1]
    filename = os.path.join(os.path.dirname(__file__), BL_NAME)
    ocr_text = test_pyocr(filename)
    for pageidx, page in enumerate(ocr_text):
        lines = get_paragraphs(page)
        for idx, line in enumerate(lines):
            print(f"[{pageidx}][{idx}] ", end="")
            print("<=====>".join([par.content for par in line]), end="")
            print("")
    ocr_text = test_pyocr(filename)
    parsed_text = parse_bl(ocr_text)
    print(parsed_text)
