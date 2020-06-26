import aiogram
from config import TOKEN
import time
import asyncio

bot = aiogram.Bot(token=TOKEN)
dp = aiogram.Dispatcher(bot)

@dp.message_handler(commands=['sleep'])
async def start_def(message: aiogram.types.Message):
    print('yee')
    f = open('images/image.jpg', 'rb')
    asyncio.sleep(5)
    await message.answer_photo(f)
    f.close()

@dp.message_handler()
async def deff(message: aiogram.types.Message):
    print('yee')
    f = open('images/image.jpg', 'rb')
    await message.answer_photo(f)
    f.close()

if __name__ == '__main__':
    aiogram.executor.start_polling(dp, skip_updates=True)