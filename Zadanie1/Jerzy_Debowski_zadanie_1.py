import re
from typing import Any

import pandas as pd
from pandas import DataFrame


def process_column(value) -> Any:
    regex_pattern = r'^[0-9]*([,.][0-9]+)?%?$'
    return value if value and re.match(regex_pattern, value) else None


class FileReader(object):

    def __init__(self, file_name: str, delimiter: str) -> None:
        df = pd.read_table(file_name, delimiter=delimiter)
        self.values = DataFrame(df)

    def _percentage_to_number(self) -> None:
        for column in self.values.columns:
            if column not in ['rok', 'miesiąc']:
                self.values[column] = self.values[column].str.replace(',', '.').str.replace('%', '').astype(float) / 100

    def _month_name_to_month_number(self) -> None:
        month_dict = {
            'styczeń': 1,
            'luty': 2,
            'marzec': 3,
            'kwiecień': 4,
            'maj': 5,
            'czerwiec': 6,
            'lipiec': 7,
            'sierpień': 8,
            'wrzesień': 9,
            'październik': 10,
            'listopad': 11,
            'grudzień': 12,
        }
        self.values['miesiąc'] = self.values['miesiąc'].str.casefold().map(month_dict)

    def _fill_na_values(self) -> None:
        for column in self.values.columns:
            if column not in ['rok', 'miesiąc']:
                mean = self.values[column].mean()
                self.values[column].fillna(mean, inplace=True)

    def _check_values(self) -> None:
        for column in self.values.columns:
            if column not in ['rok', 'miesiąc']:
                self.values[column] = self.values[column].astype(str).apply(process_column)

    def clean_values(self) -> None:
        self.values = self.values.dropna(subset=['rok', 'miesiąc'])
        self._check_values()
        self._percentage_to_number()
        self._month_name_to_month_number()
        self._fill_na_values()


file = FileReader('Punktualność_pasażerska_przewoźnicy_w_2022_r._csv. (1).csv', ';')
print(file.values[:5])
file.clean_values()
print(file.values[:5])
