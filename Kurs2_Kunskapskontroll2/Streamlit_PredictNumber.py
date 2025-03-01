
# =============================================================================
# "Pythonprogrammering och AI-utveckling - Kunskapskontroll 2 Del 1"
# Created by Anita Jonsson.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import streamlit as st

import tkinter as tk
from tkinter import filedialog
from tkinter import simpledialog
from PIL import Image, ImageDraw

from skimage import color
from skimage.transform import rescale, resize, downscale_local_mean

from numpy import asarray
import joblib
# =============================================================================
# Open window to draw figure. #
# =============================================================================
class GUI():
 
#class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="white")
        self.canvas.pack()

        # Initialize drawing state
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.is_drawing = False
        self.prev_x = None
        self.prev_y = None

        # Initialize a blank image to draw on
        self.image = Image.new('RGB', (280, 280), (255, 255, 255))  # White background
        self.draw_image = ImageDraw.Draw(self.image)

        # Button to save the image
        self.save_button = tk.Button(self.root, text="Save Image", command=self.save_image)
        self.save_button.pack()

        # Button to close the application
        self.close_button = tk.Button(self.root, text="Close", command=self.close_app)
        self.close_button.pack()

        
    def start_drawing(self, event):
        self.is_drawing = True
        self.prev_x = event.x
        self.prev_y = event.y

    def stop_drawing(self, event):
        self.is_drawing = False

    def draw(self, event):
        if self.is_drawing:
            # Draw on the canvas and on the internal image
            self.canvas.create_line(self.prev_x, self.prev_y, event.x, event.y, width=15, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
            self.draw_image.line([self.prev_x, self.prev_y, event.x, event.y], fill="black", width=15)

            self.prev_x = event.x
            self.prev_y = event.y

    def save_image(self):
        # Ask the user where to save the image
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")])
        
        if file_path:
            # Save the image
            self.image.save(file_path)
            st.header(f"Image saved as {file_path}")
        
    def close_app(self):
        # Close the window
        self.root.quit()
        
    def run(self):
        self.root.mainloop()

# =============================================================================
# Rescale picture. #
# =============================================================================

# Function to open an image file and return it as a grayscale numpy array
def open_image(uploaded_file):
    # Open the image using PIL from the uploaded file
    image = Image.open(uploaded_file)
    
    # Convert the image to grayscale
    image_gray = image.convert("L")
    
    # Convert the grayscale image to a numpy array
    image_array = np.array(image_gray)
    return image_array

# Main function to handle image transformations and plotting
def main_image(main_uploaded_file):
    # Open the image file
    image = open_image(main_uploaded_file)
    print(type(image))
    print(image.shape)
    
    if image is not None:
        # Perform the transformations on the image
        image_rescaled = rescale(image, 0.10, anti_aliasing=False)
       
        # Create a 2x2 plot grid to display the images
        fig, axes = plt.subplots(nrows=1, ncols=2)
        ax = axes.ravel()
        
        st.header(f"The picture is scaled down from 280x280 to 28x28 picture")
             
        # Display the images in the subplots
        ax[0].imshow(image, cmap='gray')
        ax[0].set_title("Original image")
        
        ax[1].imshow(image_rescaled, cmap='gray')
        ax[1].set_title("Rescaled image (aliasing)")

        # Set the limits of the first plot (optional)
        ax[0].set_xlim(0, 280)
        ax[0].set_ylim(280, 0)
        
        plt.tight_layout()
        st.pyplot(fig)

        # load the saved model
        model_name = 'model_mnist.joblib'
        mod_extra_trees = joblib.load(model_name)
      
        # Perform the transformations on the image
        image_flatten = np.round(255.0 -  (image_rescaled.reshape(1, -1) *255.0))
        image_flatten_int = image_flatten.astype(int)

        st.header(f"Prediction")
        # use the loaded model to make predictions
        predictions = mod_extra_trees.predict(image_flatten_int)
        st.header(f"The {model_name} predicted that the number in the image is: {predictions}")

# =============================================================================
# Creating the streamlit application. #
# =============================================================================

# Creating a navigation menu with three different sections the user can choose. 
nav = st.sidebar.radio("Menu",["Description", "1 Write a number", "2 Scale image and predict"])

if nav == "Description":
    st.write("Pythonprogrammering och AI-utveckling - Kunskapskontroll 2 Del 1")
    st.title("Predict handwritten numbers")
    st.write(" ")
    st.header("1 Draw a number")
    st.write("A small pop-up window is opened. ")
    st.write("Use you mouse or pad and draw a number.")
    st.write(" ")
    st.header("2 Scale image and predict ")
    st.write("The image is scaled down from 280x280 to 28x28")
    st.write("The model will predict what number you have drawn.")
    

if nav == "1 Write a number":
    st.title("Write a number")
    st.write('A small pop-up window is opened. ')
    st.write('Use you mous or pad and write a number.')
    st.write('Try to center you number in the window and try using the whole window.')
    if __name__ == "__main__":

        # Create a Tk root window and pass it to the GUI class
        root = tk.Tk()
        app = GUI(root)
        app.run()

    
if nav == "2 Scale image and predict":
    st.title("Scale image and predict")
    
    # Streamlit UI
    st.header("Select a file")
    # Allow the user to upload an image file
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "gif"])
    if uploaded_file is not None:
        # Call the main function if the user uploads a file
        main_image(uploaded_file)

