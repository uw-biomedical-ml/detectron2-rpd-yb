@echo on
:: %1 = name of dataset, %2 = path to csv
call %UserProfile%\Miniconda3-d\Scripts\activate.bat
python run_program.py %1 %2 --bm --bmo --im --ptid --imgid
echo "Program finished running!"
pause