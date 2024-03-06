import Mecab

def get_stopWords():
    # 불용어 파일 경로 확인-------------------------------------------------------------------------------------------------------------------
    stop_words = open("/home/joe/amr_ws/eda/data/koreanStopwords.txt", "r")
    lines = stop_words.readlines()
    stop_word = []
    for line in lines:
        line = line.replace("\n", "")
        stop_word.append(line)
    stop_words.close()
    return stop_word


def get_tokenized_article(refined_article):
    stop_words = get_stopWords()
    m = Mecab()
    tokenized = []

    for row in refined_article:
        inner = []
        text = m.morphs(row)
        
        for word in text:
            
            
            if (len(word) == 1) or (word in stop_words):
                pass
            else:
                inner.append(word)

        tokenized.append(inner)
    
    return tokenized

def main():
    pass

if __name__ == "__main__":
    main()