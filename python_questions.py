'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Piazza.
Use the regular expression tools built into Python; do NOT use bash.
'''

import re

def check_for_foo_or_bar(text):
    if re.search('foo', text) is None:
        print('FALSE')
    else:
        if re.search('bar', text) is None:
            print('FALSE')
        else:
            print('TRUE')
   
   '''Checks whether the input string meets the following condition.

   The string must have both the word 'foo' and the word 'bar' in it,
   whitespace- or punctuation-delimited from other words.
   (not, e.g., words like 'foobar' or 'bart' that merely contain
    the word 'bar');

   See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#match-objects

   Return:
     True if the condition is met, false otherwise.
   '''



def replace_rgb(text):
    hex = '(#([0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]|[0-9A-Fa-f][0-9A-Fa-f][0-9A-Fa-f]))'
    num = '\d+(\.)?(\d+)?'
    rgb = 'rgb(' + num + ',\s?' + num + ',\s?' + num + ')'
    color = (hex|rgb)
    output = re.sub(color, 'COLOR', text)
    return output

'''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
   '''


def edit_distance(str1, str2):
    m, n = len(str1), len(str2)
    ans = [[0 for i in range(n + 1)] for j in range(m + 1)]
    for i in range(m + 1):
        ans[i][n] = m - i
    for i in range(n + 1):
        ans[m][i] = n - i
    m -= 1
    n -= 1
    while m >= 0:
        t = n
        while t >= 0:
            if str1[m] == str2[t]:
                ans[m][t] = ans[m + 1][t + 1]
            else:
                ans[m][t] = min(ans[m][t + 1], ans[m + 1][t], ans[m + 1][t + 1]) + 1
            t -= 1
        m -= 1
    return ans[0][0]


def wine_text_processing(wine_file_path, stopwords_file_path):
    #1:
    str_count=dict()
    with open(wine_file_path) as wf:
        for line in wf:
            words=line.split()
            for word in words:
                if word in str_count:
                    str_count[word]+=1
                else:
                    str_count[word]=1

    print ('******', '\t', str_count['******'])
    print ('*****', '\t', str_count['*****'])
    print ('****', '\t', str_count['****'])
    print ('***', '\t', str_count['***'])
    print ('**', '\t', str_count['**'])
    print ('*', '\t', str_count['*'])

    print ('\n')

    #2:
    i=1
    for key in sorted(str_count, key=str_count.get, reverse=True):
        value=str_count[key]
        re.match(r'\*{1,6}', key):
            i+=1
            print (key, '\t', value + '\n')
        if i>10:
            break


    #3:
    print (str_count['a'] + '\n')

    #4:
    print (str_count['fruit'] + '\n')
    
    #5:
    print (str_count['mineral'] + '\n')
    
    #6:
    str_count_2=dict()
    for key in str_count.keys():
        if not key.lower() in str_count_2:
            str_count_2[key.lower()]=str_count[key]
        else:
            str_count_2[key.lower()]+=str_count[key]

    with open(stopwords_file_path) as sf:
        for line in sf:
            word=re.sub(r'\n$','',line)
            str_count_2.pop(word,None)

    i=1
    for key in sorted(str_count_2, key=str_count_2.get, reverse=True):
        value=str_count_2[key]
        re.match(r'\*{1,6}', key):
            i+=1
            print (key, ' ', value)
        if iterations>10:
            break


    print ('\n')
    
    #7:
    str_count=dict()
    with open(wine_file_path) as wf:
            for line in wf:
                if re.search('[*****]$', line):
                words=line.split()
                for word in words:
                    if word in str_count:
                        str_count[word]+=1
                    else:
                        str_count[word]=1

    str_count_2=dict()
    for key in str_count.keys():
        if not key.lower() in str_count_2:
            str_count_2[key.lower()]=str_count[key]
        else:
            str_count_2[key.lower()]+=str_count[key]

    with open(stopwords_file_path) as sf:
        for line in sf:
            word=re.sub(r'\n$','',line)
            str_count_2.pop(word,None)

    i=1
    for key in sorted(str_count_2,key=str_count_2.get,reverse=True):
        value=str_count_2[key]
        re.match(r'[*****]$', key):
            i+=1
            print (key, ' ', value)
        if i>10:
            break


    print ('\n')
    
    #8:
    str_count=dict()
    with open(wine_file_path) as wf:
            for line in wf:
                if re.search('*$', line):
                words=line.split()
                for word in words:
                    if word in str_count:
                        str_count[word]+=1
                    else:
                        str_count[word]=1

    str_count_2=dict()
    for key in str_count.keys():
        if not key.lower() in str_count_2:
            str_count_2[key.lower()]=str_count[key]
        else:
            str_count_2[key.lower()]+=str_count[key]

    with open(stopwords_file_path) as sf:
        for line in sf:
            word=re.sub(r'\n$','',line)
            str_count_2.pop(word,None)

    i=1
    for key in sorted(str_count_2,key=str_count_2.get,reverse=True):
        value=str_count_2[key]
        re.match(r'*$', key):
            i+=1
            print (key, '\t', value)
        if i>10:
            break


    print ('\n')
 


    

'''Process the two files to answer the following questions and output results to out.

  1. What is the distribution over star ratings?
  2. What are the 10 most common words used across all of the reviews, and how many times
     is each used?
  3. How many times does the word 'a' appear?
  4. How many times does the word 'fruit' appear?
  5. How many times does the word 'mineral' appear?
  6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
     In natural language processing, we call these common words "stop words" and often
     remove them before we process text. stopwords.txt gives you a list of some very
     common words. Remove these stopwords from your reviews. Also, try converting all the
     words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
     different words). Now what are the 10 most common words across all of the reviews,
     and how many times is each used?
  7. You should continue to use the preprocessed reviews for the following questions
     (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
     reviews, and how many times is each used? 
  8. What are the 10 most used words among the 1 star reviews, and how many times is
     each used? 
  9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
     "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
     "white" reviews?
  10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
      reviews?

  No return value.
  '''


