# Luigi pipeline
#### Выполнил Алексей Сейкин

`python -m luigi_seikin CleanupProjectTask --data-dir 'data' --dataset-series 'GSE68nnn' --dataset-name 'GSE68849' --local-scheduler`

* Класс DownloadDataset

Скачивает набор данных в формате tar-архива.

  * data_dir - параметр, указывающий директорию для сохранения данных.
  * dataset_series и dataset_name - параметры, определяющие серию и имя датасета.
  * Метод output возвращает путь к файлу для сохранения скачанных данных.
  * Метод run выполняет скачивание датасета по url. Если статус ответа сервера 200, то содержимое записывается в файл, иначе возбуждается исключение.

* Класс UnpackTarFiles

Распаковывает tar-архивы с данными.

  * Метод requires указывает, что перед выполнением этой задачи должна быть выполнена задача DownloadDataset.
  * Метод output возвращает путь к директории, куда будут распакованы файлы.
  * Метод run производит распаковку с помощью модуля tarfile и, если файл сжат gzip'ом, распаковывает его.

* Класс ProcessTextFiles

Обрабатывает текстовые файлы после распаковки.

  * Аналогично, метод requires указывает на зависимость от UnpackTarFiles.
  * Метод output возвращает путь к директории с обработанными файлами.
  * етод run читает файлы, выбирает ключи для записи в DataFrame и сохраняет их в формате TSV.

* Класс ReduceProbesTask

Уменьшает размер файла с пробами, удаляя указанные колонки.

  * Метод requires зависит от выполнения ProcessTextFiles.
  * Метод output возвращает путь к файлу с уменьшенным набором данных.
  * Метод run читает файл проб, удаляет ненужные столбцы и сохраняет результат.



