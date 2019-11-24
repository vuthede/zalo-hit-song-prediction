import ast
import os


def load_n_grams(file_path):
    """
    Load n grams from text files
    :param file_path: input to bi-gram or tri-gram file
    :return: n-gram words. E.g. bi-gram words or tri-gram words
    """
    with open(file_path, encoding="utf8") as fr:
        words = fr.read()
        words = ast.literal_eval(words)
    return words


def clean_html(html):
    """
    Clean html tags
    :param html: html text
    :return: cleaned text
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html)
    text = soup.get_text()
    return text


def clean_html_file(input_path, output_path):
    """
    Clean html tags in file and write to a new file
    :param input_path: input crawled html file
    :param output_path: path to write output content
    :return: None
    """
    if os.path.exists(output_path):
        raise Exception("Output path existed")
    with open(input_path, 'r') as fr:
        html = fr.read()
        text = clean_html(html)

    lines = text.split('\n')
    with open(output_path, 'w') as fw:
        for line in lines:
            if len(line.strip()) > 0:
                fw.write(line + "\n")


def clean_files_from_dir(input_dir, output_dir):
    """
    Clean html tags for files in a directory
    :param input_dir: path to directory
    :param output_dir: path to output director
    :return: None
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    input_files = os.listdir(input_dir)
    for input_file in input_files:
        input_file_path = os.path.join(input_dir, input_file)
        if input_file.startswith('.') or os.path.isdir(input_file_path):
            continue
        output_file_path = os.path.join(output_dir, input_file)
        clean_html_file(input_file_path, output_file_path)


"""Tests"""


def test_clean_file():
    data_path = '../data/tokenized/samples/html/html_data.txt'
    output_path = '../data/tokenized/samples/training/data.txt'
    clean_html_file(data_path, output_path)


def test_clean_files_in_dir():
    input_dir = '../data/tokenized/real/html'
    output_dir = '../data/tokenized/real/training'
    clean_files_from_dir(input_dir, output_dir)


if __name__ == '__main__':
    # test_clean_file()
    test_clean_files_in_dir()
