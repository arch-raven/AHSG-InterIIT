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
 'Yezz',
 'Yota',
 'ZTE',
 'alcatel',
 'i-mate',
 'i-mobile',
 'iNQ',
 'verykool',
 'vivo',
 'एसर',
 'एसस',
 'बेनक्यू',
 'बेनक्यू-सीमेंस',
 'ब्लैकबेरी',
 'बॉश',
 'कैसियो',
 'सेल्कॉन',
 'कूलपैड',
 'डेल',
 'एम्पोरिया',
 'एरिक्सन',
 'फुजित्सु',
 'गीगाबाइट',
 'जिओनी',
 'गूगल',
 'एचपी',
 'एचटीसी',
 'हायर',
 'हुवाई',
 'इनफिनिक्स',
 'इनोस्ट्रीम',
 'इंटेक्स',
 'इटेल',
 'कार्बन',
 'एलजी',
 'लावा',
 'लेनोवो',
 'माइक्रोमैक्स',
 'माइक्रोसॉफ्ट',
 'मित्सुबिशी',
 'मोटो',
 'मोटोरोला',
 'नोकिया',
 'वनप्लस',
 'ओप्पो',
 'पैनासोनिक',
 'फिलिप्स',
 'पोको',
 'प्रतिष्ठा',
 'रेज़र',
 'सैमसंग',
 'सीमेंस',
 'सोनी',
 'टी मोबाइल',
 'तोशीबा',
 'XOLO',
 'Xgody',
 'श्याओमी',
 'आई-मोबाइल',
 'विवो',
 'वीवो '
 'रियलमी',
 'शाओमी'}

 
brand_list_sp = list(' '+w.lower()+' ' for w  in brands)
brand_list_sp += list(' '+w.lower()+'.' for w  in brands)
brand_list_sp += list(w.lower()+' ' for w  in brands)
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

def get_brands(texts):
    '''
    A function that takes a list of strings and returns a list of brands in each string
    texts: list of strings
    '''
    brandlists = [] 
    for text in tqdm(texts):
        brandlist = _get_brands(text)
        brandlists.append(brandlist)
    return brandlists
