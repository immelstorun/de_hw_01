# python -m luigi_seikin CleanupProjectTask --data-dir 'data' --dataset-series 'GSE68nnn' --dataset-name 'GSE68849' --local-scheduler

import luigi
import requests
import os
import tarfile
import gzip
import shutil
import pandas as pd
import io
import logging

logger = logging.getLogger('luigi-interface')

# Шаг 1: Задача на скачивание данных
class DownloadDataset(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_series = luigi.Parameter(default='GSE68nnn')
    dataset_name = luigi.Parameter(default='GSE68849')

    def output(self):
        # Определяем путь для сохранения скачанного tar-архива
        return luigi.LocalTarget(os.path.join(self.data_dir, f"{self.dataset_name}_RAW.tar"))

    def run(self):
        # Создаем директорию для скачивания, если она не существует
        os.makedirs(self.data_dir, exist_ok=True)
        # Формируем URL для скачивания tar-архива
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{self.dataset_series}/{self.dataset_name}/suppl/{self.dataset_name}_RAW.tar"
        # Скачиваем tar-архив
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(self.output().path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download file with status code {response.status_code}")

    def complete(self):
        # Проверяем, что скачался непустой tar-архив
        checking_file_path = self.output().path
        if os.path.isfile(checking_file_path):
            checking_file_size = os.stat(checking_file_path).st_size
            if os.path.splitext(checking_file_path)[1] == '.tar' and checking_file_size != 0:
                logger.info(f'Размер скачанного датасета: {checking_file_size} байт')
                return True
        return False

# Шаг 2a: Задача на распаковку tar-фрхива, извлечение данных
class UnpackTarFiles(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_series = luigi.Parameter(default='GSE68nnn')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Эта задача зависит от задачи скачивания
        return DownloadDataset(dataset_name=self.dataset_name, data_dir=self.data_dir, dataset_series=self.dataset_series)

    def output(self):
        # Создаем выходной файл, содержащий пути извлеченных txt-файлов
        tmp_file = str(os.path.join(self.data_dir, self.dataset_name)) + '/tmp.txt'
        return luigi.LocalTarget(tmp_file)

    def run(self):
        # Распаковка архивов
        tar_path = self.input().path  # Путь к скачанному tar-архиву
        extract_path = os.path.join(self.data_dir, self.dataset_name)  # Путь для распаковки
        os.makedirs(extract_path, exist_ok=True)  # Создаем директорию, если она не существует

        # Распаковка tar-архива
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                file_name = os.path.splitext(member.name)[0]
                member_dir = os.path.join(extract_path, file_name)
                os.makedirs(member_dir, exist_ok=True)
                tar.extract(member, member_dir)

                # Если вложение -- gz-архив, то распаковываем в этой же папке
                if member.name.endswith('.gz'):
                    with gzip.open(os.path.join(member_dir, member.name), 'rb') as f_in:
                        with open(os.path.join(member_dir, file_name), 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(os.path.join(member_dir, member.name))

                    # Записываем пути извлеченных txt-файлов во временный текстовый файл
                    with open(self.output().path, 'a') as f:
                        f.write(os.path.join(member_dir, file_name) + '\n')
                        f.close()

    def complete(self):
        checking_file_path = self.output().path
        if os.path.isfile(checking_file_path):
            all_files_ok = True
            with open(checking_file_path, 'r') as f:
                for line in f:
                    extracted_file_path = line.strip()
                    if not os.path.isfile(extracted_file_path):
                        all_files_ok = False
                        break
                    extracted_file_size = os.stat(extracted_file_path).st_size
                    if os.path.splitext(extracted_file_path)[1] != '.txt' or extracted_file_size == 0:
                        all_files_ok = False
                        break
            return all_files_ok
        return False

# Шаг 2b: Обработка текстовых файлов, извлечение данных
class ProcessTextFiles(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_series = luigi.Parameter(default='GSE68nnn')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Зависит от успешной распаковки архивов
        return UnpackTarFiles(dataset_name=self.dataset_name, data_dir=self.data_dir, dataset_series=self.dataset_series)

    def output(self):
        # Создаем выходной файл, содержащий пути извлеченных tsv-файлов
        tmp_tsv_file = str(os.path.join(self.data_dir, self.dataset_name)) + '/tmp_tsv.txt'
        return luigi.LocalTarget(tmp_tsv_file)

    def run(self):
        dfs = {}  # Словарь для хранения датафреймов, созданных из разделов txt-файлов
        with open(self.input().path, 'r') as f:
            for line in f:
                extracted_file_path = line.replace('\n', '')

                with open(extracted_file_path, 'r') as f:
                    write_key = None
                    fio = io.StringIO()

                    for line in f.readlines():
                        if line.startswith('['):
                            if write_key:
                                fio.seek(0)
                                header = None if write_key == 'Heading' else 'infer'
                                dfs[write_key] = pd.read_csv(fio, sep='\t', header=header)
                            fio = io.StringIO()
                            write_key = line.strip('[]\n')
                            continue
                        if write_key:
                            fio.write(line)
                    fio.seek(0)
                    dfs[write_key] = pd.read_csv(fio, sep='\t')

                gz_dir = os.path.dirname(extracted_file_path)
                for k, v in dfs.items():
                    tsv_file_path = os.path.join(gz_dir, k + '.tsv')
                    v.to_csv(tsv_file_path, sep='\t')

                    with open(self.output().path, 'a') as f:
                        f.write(tsv_file_path + '\n')
                        f.close()

    def complete(self):
        checking_file_path = self.output().path
        if os.path.isfile(checking_file_path):
            all_files_ok = True
            with open(checking_file_path, 'r') as f:
                for line in f:
                    extracted_file_path = line.strip()
                    if not os.path.isfile(extracted_file_path):
                        all_files_ok = False
                        break
                    extracted_file_size = os.stat(extracted_file_path).st_size
                    if os.path.splitext(extracted_file_path)[1] != '.tsv' or extracted_file_size == 0:
                        all_files_ok = False
                        break
            return all_files_ok
        return False

# Шаг 3: Удаление ненужных колонок из таблицы "Probes.tsv"
class ReduceProbesTask(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_series = luigi.Parameter(default='GSE68nnn')
    dataset_name = luigi.Parameter(default='GSE68849')

    def requires(self):
        # Зависит от успешной распаковки архива
        return ProcessTextFiles(dataset_name=self.dataset_name, data_dir=self.data_dir, dataset_series=self.dataset_series)

    def output(self):
        # Определение выходного файла, который будет должен содержать пути к обработанным tsv-файлам
        tmp_tsv_file = str(os.path.join(self.data_dir, self.dataset_name)) + '/tmp_tsv.txt'
        return luigi.LocalTarget(tmp_tsv_file)

    def run(self):
        # Список колонок, которые необходимо удалить из таблицы "Probes.tsv"
        columns_to_remove = ['Definition', 'Ontology_Component', 'Ontology_Process', 'Ontology_Function', 'Synonyms', 'Obsolete_Probe_Id', 'Probe_Sequence']

        with open(self.input().path, 'r') as f:
            for line in f:
                tsv_file_path = line.replace('\n', '')
                if 'Probes.tsv' in tsv_file_path:
                    probes_path = tsv_file_path
                    df = pd.read_csv(probes_path, sep='\t')
                    df_reduced = df.drop(columns=columns_to_remove)
                    probes_reduced_path = os.path.dirname(probes_path) + '/Probes_reduced.tsv'
                    df_reduced.to_csv(probes_reduced_path, sep='\t', index=False)

                    with open(self.output().path, 'a') as f:
                        f.write(probes_reduced_path + '\n')
                        f.close()

    def complete(self):
        checking_file_path = self.output().path
        if os.path.isfile(checking_file_path):
            all_files_ok = True
            with open(checking_file_path, 'r') as f:
                for line in f:
                    extracted_file_path = line.strip()
                    if not os.path.isfile(extracted_file_path):
                        all_files_ok = False
                        break
                    extracted_file_size = os.stat(extracted_file_path).st_size
                    if os.path.splitext(extracted_file_path)[1] != '.tsv' or extracted_file_size == 0:
                        all_files_ok = False
                        break
            return all_files_ok
        return False

# Шаг 4: Очистка проекта
class CleanupProjectTask(luigi.Task):
    data_dir = luigi.Parameter(default='data')
    dataset_name = luigi.Parameter(default='GSE68849')
    dataset_series = luigi.Parameter(default='GSE68nnn')

    def requires(self):
        # Зависит от успешной распаковки архива
        return ReduceProbesTask(dataset_name=self.dataset_name, data_dir=self.data_dir, dataset_series=self.dataset_series)

    def output(self):
        # Создает файл 'readme.txt', который будет содержать информацию об удаленных и созданных файлах
        readme = str(self.data_dir) + '/readme.txt'
        return luigi.LocalTarget(readme)

    def run(self):
        created_files = []
        with open(self.input().path, 'r') as f:
            for line in f:
                created_files.append(line)

        base_path = os.path.join(self.data_dir, self.dataset_name)
        removed_files = []
        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    removed_files.append(file)

        cleanup_message = f"Удалённые файлы: {', '.join(removed_files)}"
        print(cleanup_message)

        with open(self.output().path, 'a') as f:
            f.write('Созданные файлы:' + '\n')
            for k in created_files:
                f.write(k)

            f.write('\n')
            f.write('Созданные временно и удаленные файлы:' + '\n')
            for k in removed_files:
                f.write(k + '\n')
            f.close()

    def complete(self):
        if self.output().exists():
            with open(self.output().path, 'r') as f:
                content = f.read()
                return 'Созданные файлы:' in content and 'Созданные временно и удаленные файлы:' in content
        return False

if __name__ == "__main__":
    luigi.run()