#!/usr/bin/env python
# coding: utf-8

# In[4]:


from fastai.vision.all import *
import gradio as gr


def is_cat(x): return x[0].isupper()


# In[5]:


im = PILImage.create('dog.jpeg')
im.thumbnail((192, 192))
im


# In[6]:


learn = load_learner('model.pkl')


# In[8]:


categories = ("Dog", "Cat")


def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


# In[9]:


classify_image(im)


# In[10]:


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ["dog.jpeg", "cat.jpeg", "apple.jpeg"]

intf = gr.Interface(fn=classify_image, inputs=image,
                    outputs=label, examples=examples)
intf.launch(inline=False)


# In[11]:


m = learn.model


# In[12]:


labels = learn.dls.vocab


def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
