#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def prediction(new_image):
    try:
        model = load_model('/input/pneumonia_model.h5')
        pred = model.predict(new_image)
    
        if pred[0][0] > 0.80:
            label='Pneumonia, risk=' + str(round(pred[0][0]*100,2)) + '%'
        elif pred[0][0] < 0.60:
            label='Normal, risk=' + str(r`ound(pred[0][0]*100,2)) + '%'
        else:
            label='Unsure, risk=' + str(round(pred[0][0]*100,2)) + '%'
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise   
    prediction = {'label':label,'pred':pred[0][0]}
    return prediction

