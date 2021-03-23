import re
import string
from tqdm import tqdm

brands = {'AT&T',
 'Acer',
 'Allview',
# 'Amazon',
 'Amoi',
 'Apple',
 'Archos',
 'Asus',
 'BQ',
 'BenQ',
 'BenQ-Siemens',
 'Benefon',
 'BlackBerry',
 'Blackview',
 'Bosch',
 'Casio',
 'Celkon',
 'Coolpad',
 'Dell',
 'Emporia',
 'Energizer',
 'Ericsson',
 'Fujitsu',
 'Garmin-Asus',
 'Gigabyte',
 'Gionee',
 'Google',
 'HP',
 'HTC',
 'Haier',
 'Honor',
 'Huawei',
 'Icemobile',
 'Infinix',
 'Innostream',
 'Intex',
 'Itel',
 'Jolla',
 'Karbonn',
 'Kyocera',
 'LG',
 'Lava',
 'LeEco',
 'Lenovo',
 'Lepow',
 'MWg',
 'Maxon',
 'Maxwest',
 'Meizu',
 'Micromax',
 'Microsoft',
 'Mitac',
 'Mitsubishi',
 'Moto',
 'Motorola',
 'Neonode',
 'Nokia',
 'Nvidia',
 'O2',
 'OnePlus',
 'Oppo',
 'Orange',
 'Palm',
 'Panasonic',
 'Pantech',
 'Parla',
 'Philips',
 'Plum',
 'Poco',
 'Prestigio',
 'QMobile',
 'Qtek',
 'Razer',
 'Realme',
 'Realmi',
 'Sagem',
 'Samsung',
 'Sendo',
 'Sewon',
 'Siemens',
 'Sonim',
 'Sony',
 'Spice',
 'T-Mobile',
 'TECNO',
 'Tel.Me.',
 'Telit',
 'Thuraya',
 'Toshiba',
 'Unnecto',
 'VK',
 'Vertu',
 'Vodafone',
 'WND',
 'Wiko',
 'XCute',
 'XOLO',
 'Xgody',
 'Xiaomi',
 'Redmi',
 'Yezz',
 'Yota',
 'ZTE',
 'alcatel',
 'i-mate',
 'i-mobile',
 'iNQ',
 'verykool',
 'vivo',
#  'एसर',
#  'एसस',
#  'बेनक्यू',
#  'बेनक्यू-सीमेंस',
#  'ब्लैकबेरी',
#  'बॉश',
#  'कैसियो',
#  'सेल्कॉन',
#  'कूलपैड',
#  'डेल',
#  'एम्पोरिया',
#  'एरिक्सन',
#  'फुजित्सु',
#  'गीगाबाइट',
#  'जिओनी',
#  'गूगल',
#  'एचपी',
#  'एचटीसी',
#  'हायर',
#  'हुवाई',
#  'इनफिनिक्स',
#  'इनोस्ट्रीम',
#  'इंटेक्स',
#  'इटेल',
#  'कार्बन',
#  'एलजी',
#  'लावा',
#  'लेनोवो',
#  'माइक्रोमैक्स',
#  'माइक्रोसॉफ्ट',
#  'मित्सुबिशी',
#  'मोटो',
#  'मोटोरोला',
#  'नोकिया',
#  'वनप्लस',
#  'ओप्पो',
#  'पैनासोनिक',
#  'फिलिप्स',
#  'पोको',
#  'प्रतिष्ठा',
#  'रेज़र',
#  'सैमसंग',
#  'सीमेंस',
#  'सोनी',
#  'टी मोबाइल',
#  'तोशीबा',
#  'XOLO',
#  'Xgody',
#  'श्याओमी',
#  'आई-मोबाइल',
#  'विवो',
#  'वीवो '
#  'रियलमी',
#  'शाओमी'
}

replace_dict = {'एसर':'Acer',
 'एसस':'Asus',
 'बेनक्यू':'BenQ',
 'बेनक्यू-सीमेंस':'BenQ-Siemens',
 'ब्लैकबेरी':'Blackberry',
 'बॉश':'Bosch',
 'कैसियो':'Casio',
 'सेल्कॉन':'Celkon',
 'कूलपैड':'Coolpad',
 'डेल':'Dell',
 'एम्पोरिया':'Emporia',
 'एरिक्सन':'Ericsson',
 'फुजित्सु':'Fujitsu',
 'गीगाबाइट':'Gigabyte',
 'जिओनी':'Gionee',
 'गूगल':'Google',
 'एचपी':'HP',
 'एचटीसी':'HTC',
 'हायर':'Haier',
 'हुवाई':'Huawei',
 'इनफिनिक्स':'Infinix',
 'इनोस्ट्रीम':'Innostream',
 'इंटेक्स':'Intex',
 'इटेल':'Itel',
 'कार्बन':'Karbonn',
 'एलजी':'LG',
 'लावा':'Lava',
 'लेनोवो':'Lenovo',
 'माइक्रोमैक्स':'Micromax',
 'माइक्रोसॉफ्ट':'Microsoft',
 'मित्सुबिशी':'Mitsubishi',
 'मोटो':'Moto',
 'मोटोरोला':'Motorola',
 'नोकिया':'Nokia',
 'वनप्लस':'OnePlus',
 'ओप्पो':'Oppo',
 'पैनासोनिक':'Panasonic',
 'फिलिप्स':'Philips',
 'पोको':'Poco',
 'रेज़र':'Razer',
 'सैमसंग':'Samsung',
 'सीमेंस':'Siemens',
 'सोनी':'Sony',
 'टी मोबाइल':'T-Mobile',
 'तोशीबा':'Toshiba',
 'श्याओमी':'Xiaomi',
 'आई-मोबाइल':'i-mobile',
 'विवो':'Vivo',
 'वीवो':'Vivo',
 'रियलमी':'Realme',
 'शाओमी':'Xiaomi'
}
pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(key) for key in replace_dict.keys()) + r')(?!\w)')
 
brand_list_sp = list(r'\b'+w.lower()+r'\b' for w  in brands)
brand_list_sp.append(r'\b'+'mi'+r'\b')
# brand_list_sp += list(' '+w.lower()+'.' for w  in brands)
# brand_list_sp += list(w.lower()+' ' for w  in brands)
brand_list = list(w.lower() for w  in brands)    
search_exp_sp = '|'.join(brand_list_sp)
search_exp = '|'.join(brand_list)



def _find_in_hashtags(hashtags):
    matches = []
    for htag in hashtags:
        matches += re.findall(search_exp, htag, re.IGNORECASE)
    return matches

def _get_unique_brands(brandlist):
    brandlist = [w.lower().strip() for w in brandlist]
    s = set(brandlist)
    return list(s)

def _get_brands(text):
    '''
    A function that returns the occurences of brands in a single string 
    text: string
    '''
    hashtags = re.findall(r'\B#\w*[a-zA-Z]+\w*', text)
    brands_hashtags = _find_in_hashtags(hashtags)
    brands_text = re.findall(search_exp_sp, text, re.IGNORECASE)
    brandlist = brands_hashtags + brands_text
    brandlist = _get_unique_brands(brandlist)
    return brandlist

def get_brands(texts, verbose=True):
    '''
    A function that takes a list of strings and returns a list of brands in each string
    texts: list of strings
    '''
    brandlists = [] 
    for text in (tqdm(texts) if verbose else texts):
        brandlist = _get_brands(text)
        brandlists.append(brandlist)
    return brandlists

def _replace_hin_to_eng(text):
    '''
    internal function, do not call
    '''
    result = pattern.sub(lambda x: replace_dict[x.group()], text)
    return result

def replace_hin_to_eng(texts):
    '''
    Takes a list of strings, and replace the brands in them from Devanagri -> Latin
    Input: 
    texts - an iterable of strings
    Output:
    repl_texts - an iterable of replaced strings
    '''
    repl_texts = []
    for text in texts:
        result = _replace_hin_to_eng(text)
        repl_texts.append(result)
    return repl_texts

def _get_brand_indices(text):
    brands = _get_brands(text)
    match_indices = dict()
    for brand in brands:
        occ = []
        pattern = r'\b'+brand+r'\b'
        pattern_htag = brand
        split_text = re.findall(r"[\w]+|[^\s\w]", text)
        for i,word in enumerate(split_text):
            if split_text[max(0,i-1)] == '#' and re.search(pattern_htag,word,re.IGNORECASE)!= None:
                occ.append(i)            
            elif re.search(pattern,word,re.IGNORECASE)!= None:
                occ.append(i)
        match_indices[brand] = occ
    return match_indices

def get_brand_indices(texts):
    '''
    Takes a list of strings, and returns a dict of the brand occurence indices
    Eg: 
     'Apple and Samsung are good brands but Apple is better'
    =>{'apple':[0,7],
       'samsung': [2]
      }  
    '''
    results = []
    for text in tqdm(texts):
        match_indices = _get_brand_indices(text)
        results.append(match_indices)
    return results


