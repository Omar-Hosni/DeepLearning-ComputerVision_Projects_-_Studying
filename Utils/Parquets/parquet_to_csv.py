import pandas as pd
import io
def convert(parquet_file, csv_file):
    try:
        df = pd.read_parquet(parquet_file).head(100)

        # Initialize lists to store the extracted conversation data
        person_talk = []
        helper_reply = []

        # Loop through each row in the DataFrame
        for index, row in df.iterrows():
            # Get the content of the current row
            content = row['text'].strip()

            # Split the row's content by '###'
            talk_segments = content.split('###')

            # Initialize variables to store sentences for person and helper
            person_sentence = ""
            helper_sentence = ""

            # Initialize a flag to keep track of whether the previous segment is for the person or helper
            is_person_talk = True

            # Loop through the talk segments
            for segment in talk_segments:
                # Remove any leading and trailing whitespaces from the segment
                segment = segment.strip()

                # Determine if the segment is for the person or helper
                if segment.startswith('الانسان:'):
                    person_sentence += segment[len('الانسان:'):].strip() + ' '
                    is_person_talk = True
                elif segment.startswith('المساعد:'):
                    helper_sentence += segment[len('المساعد:'):].strip() + ' '
                    is_person_talk = False
                else:
                    # If the segment does not start with 'الانسان:' or 'المساعد:',
                    # it belongs to the previous speaker
                    if is_person_talk:
                        person_sentence += segment + ' '
                    else:
                        helper_sentence += segment + ' '



            # Append the sentences to the corresponding lists
            person_talk.append(person_sentence.strip())
            helper_reply.append(helper_sentence.strip())

        with io.open(csv_file, 'w', encoding='utf-8-sig') as f:
            f.write('person, helper\n')
            for person, helper in zip(person_talk, helper_reply):
                f.write(f"{person},{helper}\n")
                print(f'person: {person} helper: {helper}')

        print('Conversion Complete')

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == '__main__':

    input_file = 'train.parquet'
    output_file = 'output_file.csv'
    convert(input_file, output_file)
    #df = pd.read_parquet('train.parquet').head(100)

    #for index,row in df.iterrows():
        #print(row[0])