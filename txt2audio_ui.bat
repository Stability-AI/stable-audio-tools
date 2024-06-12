@echo off
setlocal

set config_path='config/txt2audio.json'

for /f "delims=" %%i in ('python -c "import json; f=open(%config_path%); data=json.load(f); f.close(); print(data['model_selected'])"') do set model_selected=%%i

set models_path=.\models\

if not exist %models_path% (
    mkdir %models_path%
    echo No model found
    pause
    exit /b
)

if defined model_selected (
    for %%i in (.ckpt .safetensors .pth) do  (
        if exist %models_path%%model_selected%%%i (
            set model_path=%models_path%%model_selected%%%i
            set model_name=%model_selected%
            goto :model_found
        )
    )
)
set config_model_found=.
echo No model found in config file
echo Searching in models folder

for /R %models_path% %%f in (*.ckpt *.safetensors *.pth) do (
    set model_path=%%~dpnxf
    set model_name=%%~nf
    goto :model_found
)
echo No model found
pause
exit /b

:model_found
echo Found model: %model_name%
set model_config_path=%models_path%%model_name%.json
if not exist %model_config_path% (
    echo Model config not found.
    pause
    exit /b
)

if defined config_model_found (
    python -c "import json; f=open(%config_path%); data=json.load(f); data['model_selected'] = '%model_name%'; f.close(); f=open(%config_path%, 'w'); json.dump(data, f, indent=4); f.write('\n'); f.close()"
)

call .\venv\Scripts\activate.bat
call python run_gradio.py --ckpt-path %model_path% --model-config %model_config_path% --inbrowser
pause
