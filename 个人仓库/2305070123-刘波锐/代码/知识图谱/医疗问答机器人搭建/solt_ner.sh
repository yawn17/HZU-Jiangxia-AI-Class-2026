cd D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\liu_Medical_KG\NLU\Slot_filling
nohup gunicorn -w 1 -b 0.0.0.0:6002 solt_app:app >D:\大三上\08 正课 红蜘蛛知识图谱项目3.0-V6-25年7月版本-12天-AI版本\liu_Medical_KG\logs\solt_output.log 2>&1 &
#python solt_app.py