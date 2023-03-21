import os
import logging
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the machine learning model
model = torch.load('cifar10_model.pth', map_location=torch.device('cpu'))

# Define the image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define the command handler for the /start command
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi! Send me an image and I will tell you what is shown on it.')

# Define the message handler for image messages
def image_handler(update: Update, context: CallbackContext) -> None:
    """Process an image and send the predictions."""
    # Get the file ID of the received image
    file_id = update.message.photo[-1].file_id
    # Download the image file
    file = context.bot.get_file(file_id)
    file.download('image.jpg')
    # Preprocess the image
    image = Image.open('image.jpg').convert('RGB')
    image_tensor = preprocess(image)
    image_tensor.unsqueeze_(0)
    # Make the prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, predicted_classes = torch.max(probabilities, 1)
        predicted_class = predicted_classes.item()
    # Send the prediction
    update.message.reply_text(f'The image shows {predicted_class}.')

# Define the error handler
def error_handler(update: Update, context: CallbackContext) -> None:
    """Log the error and send a message to the user."""
    logger.warning(f'Update {update} caused error {context.error}')
    update.message.reply_text('Sorry, something went wrong.')

def main() -> None:
    """Start the bot."""
    # Create the Updater and get the bot token from the environment variable
    updater = Updater(os.environ['id_token'])
    # Set up the handlers
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))
    updater.dispatcher.add_error_handler(error_handler)
    # Start the bot
    updater.start_polling()
    logger.info('Bot started.')
    updater.idle()

if __name__ == '__main__':
    main()

#updater = Updater(token='***************************', use_context=True)
