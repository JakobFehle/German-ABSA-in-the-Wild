restaurant_aspect_cate_list = ['food', 'service', 'general impression', 'ambience', 'price']

transport_aspect_cate_list = ['general', 'miscellaneous', 'tickets', 'safety', 'atmosphere',
     'ride', 'information', 'capacity', 'app', 'service',
     'connectivity', 'child friendliness', 'comfort', 'design', 'toilets',
     'accessibility', 'luggage', 'food', 'qr code', 'image']

inclusion_aspect_cate_list = ['space', 'lift', 'lighting', 'accidents', 'barrier general',
     'escalator', 'info display', 'info', 'info acoustic', 'guiding routes', 
     'barrier others', 'ground level access', 'construction site', 'ramp', 'demonstration',
     'security', 'acoustic signal']

software_aspect_cate_list = ['ease of use', 'interface design', 'general experience', 'pricing', 'customer support', 'technical performance', 'functional scope']

POLARITY_MAPPING_POL_TO_TERM = {"negative": "schlecht", "neutral": "okay", "positive": "gut"}

cate_list = {
    "restaurant": restaurant_aspect_cate_list,
    "transport": transport_aspect_cate_list,
    "inclusion": inclusion_aspect_cate_list,
    "software/v2": software_aspect_cate_list
}

task_data_list = {
    "tasd": ["restaurant", "hotel", "transport", "inclusion", "software/v2"],
}

force_words = {
    'tasd': {
        "restaurant": restaurant_aspect_cate_list + list(POLARITY_MAPPING_POL_TO_TERM.values()) + ['[SSEP]'],
        "transport": transport_aspect_cate_list + list(POLARITY_MAPPING_POL_TO_TERM.values()) + ['[SSEP]'],
        "inclusion": inclusion_aspect_cate_list + list(POLARITY_MAPPING_POL_TO_TERM.values()) + ['[SSEP]'],
        "software/v2": software_aspect_cate_list + list(POLARITY_MAPPING_POL_TO_TERM.values()) + ['[SSEP]']
    }
}