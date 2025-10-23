#!/usr/bin/env python3
"""
Тест для проверки корректности настройки TensorBoard.
Проверяет, что все необходимые импорты работают и TensorBoardLogger можно создать.
"""

import sys
import os

def test_imports():
    """Проверка импортов."""
    print("Проверка импортов...")
    
    try:
        import lightning as L
        print("✓ Lightning импортирован")
    except ImportError as e:
        print(f"✗ Ошибка импорта Lightning: {e}")
        return False
    
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
        print("✓ TensorBoardLogger импортирован")
    except ImportError as e:
        print(f"✗ Ошибка импорта TensorBoardLogger: {e}")
        return False
    
    try:
        import torch
        print("✓ PyTorch импортирован")
    except ImportError as e:
        print(f"✗ Ошибка импорта PyTorch: {e}")
        return False
    
    try:
        import hydra
        import omegaconf
        print("✓ Hydra и OmegaConf импортированы")
    except ImportError as e:
        print(f"✗ Ошибка импорта Hydra/OmegaConf: {e}")
        return False
    
    return True

def test_tensorboard_logger():
    """Проверка создания TensorBoardLogger."""
    print("\nПроверка создания TensorBoardLogger...")
    
    try:
        from lightning.pytorch.loggers import TensorBoardLogger
        import tempfile
        
        # Создаем временную директорию для тестирования
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TensorBoardLogger(
                save_dir=tmpdir,
                name="test_experiment",
                version="test_version",
                default_hp_metric=False
            )
            
            print(f"✓ TensorBoardLogger создан успешно")
            print(f"  - Save dir: {logger.save_dir}")
            print(f"  - Name: {logger.name}")
            print(f"  - Version: {logger.version}")
            print(f"  - Log dir: {logger.log_dir}")
            
            # Тест логирования гиперпараметров
            logger.log_hyperparams({"learning_rate": 0.001, "batch_size": 32})
            print("✓ Гиперпараметры залогированы")
            
            # Тест логирования метрик
            logger.log_metrics({"train/loss": 0.5, "val/accuracy": 0.8}, step=0)
            print("✓ Метрики залогированы")
            
            return True
            
    except Exception as e:
        print(f"✗ Ошибка при создании TensorBoardLogger: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensorboard_installation():
    """Проверка установки TensorBoard."""
    print("\nПроверка установки TensorBoard...")
    
    try:
        import tensorboard
        print(f"✓ TensorBoard установлен (версия: {tensorboard.__version__})")
        return True
    except ImportError:
        print("✗ TensorBoard не установлен")
        print("  Установите его: pip install tensorboard")
        return False

def test_config_files():
    """Проверка конфигурационных файлов."""
    print("\nПроверка конфигурационных файлов...")
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Проверяем наличие конфигов
    config_files = [
        "configs/config.yaml",
        "configs/logging/tensorboard.yaml",
    ]
    
    all_exist = True
    for config_file in config_files:
        full_path = os.path.join(project_root, config_file)
        if os.path.exists(full_path):
            print(f"✓ {config_file} существует")
        else:
            print(f"✗ {config_file} не найден")
            all_exist = False
    
    # Проверяем скрипты
    scripts = [
        "launch_tensorboard.sh",
        "view_latest_run.sh",
        "my_scripts/train.sh",
    ]
    
    for script in scripts:
        full_path = os.path.join(project_root, script)
        if os.path.exists(full_path):
            is_executable = os.access(full_path, os.X_OK)
            status = "✓" if is_executable else "⚠"
            exec_msg = "исполняемый" if is_executable else "не исполняемый"
            print(f"{status} {script} существует ({exec_msg})")
            if not is_executable:
                print(f"   Сделайте исполняемым: chmod +x {script}")
        else:
            print(f"✗ {script} не найден")
            all_exist = False
    
    return all_exist

def main():
    """Главная функция теста."""
    print("="*60)
    print("Тест настройки TensorBoard для MDLM")
    print("="*60)
    
    results = []
    
    # Запускаем тесты
    results.append(("Импорты", test_imports()))
    results.append(("TensorBoard установка", test_tensorboard_installation()))
    results.append(("TensorBoardLogger", test_tensorboard_logger()))
    results.append(("Конфигурационные файлы", test_config_files()))
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТОВ")
    print("="*60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 Все тесты пройдены! TensorBoard настроен корректно.")
        print("\nДля запуска обучения:")
        print("  bash my_scripts/train.sh")
        print("\nДля просмотра логов:")
        print("  bash view_latest_run.sh")
        return 0
    else:
        print("\n⚠ Некоторые тесты не прошли. Проверьте ошибки выше.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

