from fastapi import FastAPI
from func import greetings
import uvicorn

tiguan = FastAPI()

@tiguan.get("/")
def read_root():
    return {"Message": greetings()}
    # return {"Message": "Hello World"}

if __name__ == '__main__':
    uvicorn.run("test:tiguan",
                host="127.0.0.1",
                port=3000,
                reload=True)

