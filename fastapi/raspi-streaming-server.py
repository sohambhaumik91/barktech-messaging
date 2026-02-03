from fastapi import FastAPI, status
from fastapi.responses import HTMLResponse
app = FastAPI()



@app.get("/", response_class=HTMLResponse)
async def show_raspi_dashboard():
    return """ 
        <!DOCTYPE html>
        <html>
            <head> 
                <title>Pi Stream</title>
                <style> 
                    body { font-family: sans-serif; text-align: center; background: #222; color: white; } 
                    img { border: 2px solid #555; background: #333; } .controls { margin-top: 20px; } 
                    button { padding: 12px 24px; cursor: pointer; } 
                </style> </head> <body> <h1>Raspberry Pi Camera</h1>
            </head>
            <body> 
                <div class="controls"> <button onclick="fetch('/cmd/on')">ON</button> <button onclick="fetch('/cmd/off')">OFF</button>
                </div> 
            </body> 
        </html> """
        
@app.get("/cmd/{cmd_type}")
async def execute_system_cmd(cmd_type: str):
    print(f"command of type: {cmd_type} has been added to the queue")
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)