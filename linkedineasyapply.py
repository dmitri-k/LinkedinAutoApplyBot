import time, random, csv, pyautogui, traceback, os, re
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from datetime import date, datetime
from itertools import product
from pypdf import PdfReader
from openai import OpenAI
import sys
import pdb  # Import the Python debugger

class AIResponseGenerator:
    def __init__(self, api_key, personal_info, experience, languages, resume_path, text_resume_path=None, debug=False):
        self.personal_info = personal_info
        self.experience = experience
        self.languages = languages
        self.pdf_resume_path = resume_path
        self.text_resume_path = text_resume_path
        self._resume_content = None
        self._client = OpenAI(api_key=api_key) if api_key else None
        self.debug = debug
    @property
    def resume_content(self):
        if self._resume_content is None:
            # First try to read from text resume if available
            if self.text_resume_path:
                try:
                    with open(self.text_resume_path, 'r', encoding='utf-8') as f:
                        self._resume_content = f.read()
                        print("Successfully loaded text resume")
                        return self._resume_content
                except Exception as e:
                    print(f"Could not read text resume: {str(e)}")

            # Fall back to PDF resume if text resume fails or isn't available
            try:
                content = []
                reader = PdfReader(self.pdf_resume_path)
                for page in reader.pages:
                    content.append(page.extract_text())
                self._resume_content = "\n".join(content)
                print("Successfully loaded PDF resume")
            except Exception as e:
                print(f"Could not extract text from resume PDF: {str(e)}")
                self._resume_content = ""
        return self._resume_content

    def _build_context(self):
        return f"""
        Personal Information:
        - Name: {self.personal_info['First Name']} {self.personal_info['Last Name']}
        - Current Role: {self.experience.get('currentRole', '')}
        - Skills: {', '.join(self.experience.keys())}
        - Languages: {', '.join(f'{lang}: {level}' for lang, level in self.languages.items())}
        - Professional Summary: {self.personal_info.get('MessageToManager', '')}

        Resume Content (Give the greatest weight to this information, if specified):
        {self.resume_content}
        """

    def generate_response(self, question_text, response_type="text", options=None, max_tokens=100):
        """
        Generate a response using OpenAI's API
        
        Args:
            question_text: The application question to answer
            response_type: "text", "numeric", or "choice"
            options: For "choice" type, a list of tuples containing (index, text) of possible answers
            max_tokens: Maximum length of response
            
        Returns:
            - For text: Generated text response or None
            - For numeric: Integer value or None
            - For choice: Integer index of selected option or None
        """
        if not self._client:
            return None
            
        try:
            context = self._build_context()
            
            system_prompt = {
                "text": "You are a helpful assistant answering job application questions professionally and concisely. Use the candidate's background information and resume to personalize responses.",
                "numeric": "You are a helpful assistant providing numeric answers to job application questions. Based on the candidate's experience, provide a single number as your response. No explanation needed.",
                "choice": "You are a bot helpfull auto applying for jobs and must select the most appropriate answer choice for job application questions. Based on the candidate's background, select the best option by returning only its index number. No explanation needed. If necessary make a guess of most likely answer. "
            }[response_type]

            user_content = f"Using this candidate's background and resume:\n{context}\n\n Please answer this job application question: {question_text}\n\n It is 2025 and candidate has 6 years experience of software so reference that if hesistant on anything. Lean a bit on the embelishment side so for front end technologies say 5+ years experience for example."
            if response_type == "choice" and options:
                options_text = "\n".join([f"{idx}: {text}" for idx, text in options])
                user_content += f"\n\nSelect the most appropriate answer by providing its index number from these options:\n{options_text}"

            response = self._client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"AI response: {answer}")  # TODO: Put logging behind a debug flag
            
            if response_type == "numeric":
                # Extract first number from response
                numbers = re.findall(r'\d+', answer)
                if numbers:
                    return int(numbers[0])
                return 0
            elif response_type == "choice":
                # Extract the index number from the response
                numbers = re.findall(r'\d+', answer)
                if numbers and options:
                    index = int(numbers[0])
                    # Ensure index is within valid range
                    if 0 <= index < len(options):
                        return index
                return None  # Return None if the index is not within the valid range
                
            return answer
            
        except Exception as e:
            print(f"Error using AI to generate response: {str(e)}")
            return None

    def evaluate_job_fit(self, job_title, job_description):
        """
        Evaluate whether a job is worth applying to based on the candidate's experience and the job requirements
        
        Args:
            job_title: The title of the job posting
            job_description: The full job description text
            
        Returns:
            bool: True if should apply, False if should skip
        """
        if not self._client:
            return True  # Proceed with application if AI not available
            
        try:
            context = self._build_context()
            
            system_prompt = """You are evaluating job fit for technical roles. 
            Recommend APPLY if:
            - Candidate meets 65 percent of the core requirements
            - Experience gap is 2 years or less
            - Has relevant transferable skills
            
            Return SKIP if:
            - Experience gap is greater than 2 years
            - Missing multiple core requirements
            - Role is clearly more senior
            - The role is focused on an uncommon technology or skill that is required and that the candidate does not have experience with
            - The role is a leadership role or a role that requires managing people and the candidate has no experience leading or managing people

            """
            #Consider the candidate's education level when evaluating whether they meet the core requirements. Having higher education than required should allow for greater flexibility in the required experience.
            
            if self.debug:
                system_prompt += """
                You are in debug mode. Return a detailed explanation of your reasoning for each requirement.

                Return APPLY or SKIP followed by a brief explanation.

                Format response as: APPLY/SKIP: [brief reason]"""
            else:
                system_prompt += """Return only APPLY or SKIP."""

            response = self._client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Job: {job_title}\n{job_description}\n\nCandidate:\n{context}"}
                ],
                max_tokens=250 if self.debug else 1,  # Allow more tokens when debug is enabled
                temperature=0.2  # Lower temperature for more consistent decisions
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"AI evaluation: {answer}")
            return answer.upper().startswith('A')  # True for APPLY, False for SKIP
            
        except Exception as e:
            print(f"Error evaluating job fit: {str(e)}")
            return True  # Proceed with application if evaluation fails

class LinkedinEasyApply:
    def __init__(self, parameters, driver):
        self.browser = driver
        self.email = parameters['email']
        self.email = parameters['lastName']
        self.password = parameters['password']
        self.openai_api_key = parameters.get('openaiApiKey', '')  # Get API key with empty default
        self.disable_lock = parameters['disableAntiLock']
        self.company_blacklist = parameters.get('companyBlacklist', []) or []
        self.title_blacklist = parameters.get('titleBlacklist', []) or []
        self.poster_blacklist = parameters.get('posterBlacklist', []) or []
        self.positions = parameters.get('positions', [])
        self.locations = parameters.get('locations', [])
        self.residency = parameters.get('residentStatus', [])
        self.base_search_url = self.get_base_search_url(parameters)
        self.seen_jobs = []
        self.file_name = "output"
        self.unprepared_questions_file_name = "unprepared_questions"
        self.output_file_directory = parameters['outputFileDirectory']
        self.resume_dir = parameters['uploads']['resume']
        self.text_resume = parameters.get('textResume', '')
        if 'coverLetter' in parameters['uploads']:
            self.cover_letter_dir = parameters['uploads']['coverLetter']
        else:
            self.cover_letter_dir = ''
        self.checkboxes = parameters.get('checkboxes', [])
        self.university_gpa = parameters['universityGpa']
        self.salary_minimum = parameters['salaryMinimum']
        self.notice_period = int(parameters['noticePeriod'])
        self.languages = parameters.get('languages', [])
        self.experience = parameters.get('experience', [])
        self.personal_info = parameters.get('personalInfo', [])
        self.eeo = parameters.get('eeo', [])
        self.experience_default = int(self.experience['default'])
        self.debug = parameters.get('debug', False)
        self.evaluate_job_fit = parameters.get('evaluateJobFit', True)
        self.ai_response_generator = AIResponseGenerator(
            api_key=self.openai_api_key,
            personal_info=self.personal_info,
            experience=self.experience,
            languages=self.languages,
            resume_path=self.resume_dir,
            text_resume_path=self.text_resume,
            debug=self.debug
        )

    def login(self):
        try:
            # Check if the "chrome_bot" directory exists
            print("Attempting to restore previous session...")
            if os.path.exists("chrome_bot"):
                self.browser.get("https://www.linkedin.com/feed/")
                time.sleep(random.uniform(5, 10))

                # Check if the current URL is the feed page
                if self.browser.current_url != "https://www.linkedin.com/feed/":
                    print("Feed page not loaded, proceeding to login.")
                    self.load_login_page_and_login()
            else:
                print("No session found, proceeding to login.")
                self.load_login_page_and_login()

        except TimeoutException:
            print("Timeout occurred, checking for security challenges...")
            self.security_check()
            # raise Exception("Could not login!")

    def security_check(self):
        current_url = self.browser.current_url
        page_source = self.browser.page_source

        if '/checkpoint/challenge/' in current_url or 'security check' in page_source or 'quick verification' in page_source:
            input("Please complete the security check and press enter on this console when it is done.")
            time.sleep(random.uniform(5.5, 10.5))

    def load_login_page_and_login(self):
        self.browser.get("https://www.linkedin.com/login")

        # Wait for the username field to be present
        WebDriverWait(self.browser, 10).until(
            EC.presence_of_element_located((By.ID, "username"))
        )

        self.browser.find_element(By.ID, "username").send_keys(self.email)
        self.browser.find_element(By.ID, "password").send_keys(self.password)
        self.browser.find_element(By.CSS_SELECTOR, ".btn__primary--large").click()

        # Wait for the feed page to load after login
        WebDriverWait(self.browser, 10).until(
            EC.url_contains("https://www.linkedin.com/feed/")
        )

        time.sleep(random.uniform(5, 10))

    def start_applying(self):
        searches = list(product(self.positions, self.locations))
        random.shuffle(searches)

        page_sleep = 0
        minimum_time = 60 * 2  # minimum time bot should run before taking a break
        minimum_page_time = time.time() + minimum_time

        for (position, location) in searches:
            location_url = "&location=" + location
            job_page_number = -1

            print("Starting the search for " + position + " in " + location + ".")

            try:
                while True:
                    page_sleep += 1
                    job_page_number += 1
                    print("Going to job page " + str(job_page_number))
                    self.next_job_page(position, location_url, job_page_number)
                    time.sleep(random.uniform(1.5, 3.5))
                    print("Starting the application process for this page...")
                    self.apply_jobs(location)
                    print("Job applications on this page have been successfully completed.")

                    time_left = minimum_page_time - time.time()
                    if time_left > 0:
                        print("Sleeping for " + str(time_left) + " seconds.")
                        time.sleep(time_left)
                        minimum_page_time = time.time() + minimum_time
                    if page_sleep % 5 == 0:
                        sleep_time = random.randint(180, 300)  # Changed from 500, 900 {seconds}
                        print("Sleeping for " + str(sleep_time / 60) + " minutes.")
                        time.sleep(sleep_time)
                        page_sleep += 1
            except:
                traceback.print_exc()
                pass

            time_left = minimum_page_time - time.time()
            if time_left > 0:
                print("Sleeping for " + str(time_left) + " seconds.")
                time.sleep(time_left)
                minimum_page_time = time.time() + minimum_time
            if page_sleep % 5 == 0:
                sleep_time = random.randint(500, 900)
                print("Sleeping for " + str(sleep_time / 60) + " minutes.")
                time.sleep(sleep_time)
                page_sleep += 1

    def apply_jobs(self, location):
        no_jobs_text = ""
        try:
            no_jobs_element = self.browser.find_element(By.CLASS_NAME, 'jobs-search-two-pane__no-results-banner--expand')
            no_jobs_text = no_jobs_element.text
        except:
            pass
        if 'No matching jobs found' in no_jobs_text:
            raise Exception("No more jobs on this page.")

        if 'unfortunately, things are' in self.browser.page_source.lower():
            raise Exception("No more jobs on this page.")

        job_results_header = ""
        maybe_jobs_crap = ""
        try:
            job_results_header = self.browser.find_element(By.CLASS_NAME, "jobs-search-results-list__text")
            maybe_jobs_crap = job_results_header.text
        except:
            pass

        if 'Jobs you may be interested in' in maybe_jobs_crap:
            raise Exception("Nothing to do here, moving forward...")

        processed_jobs = 0
        while True:
            try:
                # Re-fetch job list dynamically
                ul_element_class = self.get_job_list_class()  # Get the current UL class
                WebDriverWait(self.browser, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, ul_element_class))
                )
                job_list = self.browser.find_elements(By.CLASS_NAME, ul_element_class)[0].find_elements(
                    By.CLASS_NAME, 'scaffold-layout__list-item'
                )
                print(f"Found {len(job_list)} jobs on this page")

                if processed_jobs >= len(job_list):
                    print("Processed all jobs on this page.")
                    break

                job_tile = job_list[processed_jobs]
                print(job_tile)

                # Extract job details
                job_title, company, poster, job_location, apply_method, link = "", "", "", "", "", ""
                try:
                    job_title_element = job_tile.find_element(By.CLASS_NAME, 'job-card-list__title--link')
                    job_title = job_title_element.find_element(By.TAG_NAME, 'strong').text
                    link = job_tile.find_element(By.CLASS_NAME, 'job-card-list__title--link').get_attribute('href').split('?')[0]
                except:
                    pass

                try:
                    company = job_tile.find_element(By.CLASS_NAME, 'artdeco-entity-lockup__subtitle').text
                except:
                    pass

                try:
                    hiring_line = job_tile.find_element(By.XPATH, '//span[contains(.,\' is hiring for this\')]')
                    hiring_line_text = hiring_line.text
                    name_terminating_index = hiring_line_text.find(' is hiring for this')
                    if name_terminating_index != -1:
                        poster = hiring_line_text[:name_terminating_index]
                except:
                    pass

                try:
                    job_location = job_tile.find_element(By.CLASS_NAME, 'job-card-container__metadata-item').text
                except:
                    pass

                try:
                    apply_method = job_tile.find_element(By.CLASS_NAME, 'job-card-container__apply-method').text
                except:
                    pass

                contains_blacklisted_keywords = False
                blacklisted_word_found = ""
                job_title_parsed = job_title.lower().split(' ')
                for word in self.title_blacklist:
                    if word.lower() in job_title_parsed:
                        contains_blacklisted_keywords = True
                        blacklisted_word_found = word
                        break

                if (company.lower() not in [word.lower() for word in self.company_blacklist] and
                    poster.lower() not in [word.lower() for word in self.poster_blacklist] and
                    contains_blacklisted_keywords is False):
                    try:
                        max_retries = 3
                        retries = 0
                        while retries < max_retries:
                            try:
                                job_el = job_tile.find_element(By.CLASS_NAME, 'job-card-job-posting-card-wrapper__card-link')
                                job_el.click()
                                break
                            except StaleElementReferenceException:
                                retries += 1
                                time.sleep(1)
                                continue
                        else:
                            print("Failed to click job after retries due to StaleElementReferenceException")
                            processed_jobs += 1
                            continue

                        time.sleep(random.uniform(3, 5))

                        if self.evaluate_job_fit:
                            try:
                                job_description = self.browser.find_element(By.ID, 'job-details').text
                                if not self.ai_response_generator.evaluate_job_fit(job_title, job_description):
                                    print("Skipping application: Job requirements not aligned with candidate profile per AI evaluation.")
                                    processed_jobs += 1
                                    continue
                            except:
                                print("Could not load job description")

                        try:
                            done_applying = self.apply_to_job()
                            if done_applying:
                                print(f"Application sent to {company} for the position of {job_title}.")
                            else:
                                print(f"An application for a job at {company} has been submitted earlier.")
                        except:
                            temp = self.file_name
                            self.file_name = "failed"
                            print("Failed to apply to job. Please submit a bug report with this link: " + link)
                            try:
                                self.write_to_file(company, job_title, link, job_location, location)
                            except:
                                pass
                            self.file_name = temp
                            print(f'updated {temp}.')

                        try:
                            self.write_to_file(company, job_title, link, job_location, location)
                        except Exception:
                            print(f"Unable to save the job information in the file. The job title {job_title} or company {company} cannot contain special characters,")
                            traceback.print_exc()

                        processed_jobs += 1

                    except Exception as e:
                        traceback.print_exc()
                        print(f"Could not apply to the job in {company}: {e}")
                        processed_jobs += 1
                        continue
                else:
                    reasons = []
                    if contains_blacklisted_keywords:
                        reasons.append(f"job title contains blacklisted keyword '{blacklisted_word_found}'")
                    if link in self.seen_jobs:
                        reasons.append("job has already been seen")
                    print(f"Job for {company} by {poster} skipped because " + " and ".join(reasons) + ".")
                    processed_jobs += 1

                self.seen_jobs.append(link)

            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                break
    
    def get_job_list_class(self):
        """Helper method to fetch the UL element class dynamically."""
        try:
            xpath_region1 = "/html/body/div[6]/div[3]/div[4]/div/div/main/div/div[2]/div[1]/div/ul"
            xpath_region2 = "/html/body/div[5]/div[3]/div[4]/div/div/main/div/div[2]/div[1]/div/ul"
            try:
                ul_element = self.browser.find_element(By.XPATH, xpath_region1)
            except NoSuchElementException:
                ul_element = self.browser.find_element(By.XPATH, xpath_region2)
            return ul_element.get_attribute("class").split()[0]
        except Exception as e:
            print(f"Error fetching job list class: {e}")
            raise
   
    def apply_to_job(self):
        easy_apply_button = None

        try:
            easy_apply_button = self.browser.find_element(By.CLASS_NAME, 'jobs-apply-button')
        except:
            return False

        try:
            job_description_area = self.browser.find_element(By.ID, "job-details")
            print (f"{job_description_area}")
            self.scroll_slow(job_description_area, end=1600)
            self.scroll_slow(job_description_area, end=1600, step=400, reverse=True)
        except:
            pass

        print("Starting the job application...")
        easy_apply_button.click()

        button_text = ""
        submit_application_text = 'submit application'
        while submit_application_text not in button_text.lower():
            try:
                self.fill_up()
                next_button = self.browser.find_element(By.CLASS_NAME, "artdeco-button--primary")
                button_text = next_button.text.lower()
                if submit_application_text in button_text:
                    try:
                        self.unfollow()
                    except:
                        print("Failed to unfollow company.")
                time.sleep(random.uniform(1.5, 2.5))
                next_button.click()
                time.sleep(random.uniform(3.0, 5.0))

                # Newer error handling
                error_messages = [
                    'enter a valid',
                    'enter a decimal',
                    'Enter a whole number'
                    'Enter a whole number between 0 and 99',
                    'file is required',
                    'whole number',
                    'make a selection',
                    'select checkbox to proceed',
                    'saisissez un numéro',
                    '请输入whole编号',
                    '请输入decimal编号',
                    '长度超过 0.0',
                    'Numéro de téléphone',
                    'Introduce un número de whole entre',
                    'Inserisci un numero whole compreso',
                    'Preguntas adicionales',
                    'Insira um um número',
                    'Cuántos años'
                    'use the format',
                    'A file is required',
                    '请选择',
                    '请 选 择',
                    'Inserisci',
                    'wholenummer',
                    'Wpisz liczb',
                    'zakresu od',
                    'tussen'
                ]

                if any(error in self.browser.page_source.lower() for error in error_messages):
                    raise Exception("Failed answering required questions or uploading required files.")
            except:
                traceback.print_exc()
                self.browser.find_element(By.CLASS_NAME, 'artdeco-modal__dismiss').click()
                time.sleep(random.uniform(3, 5))
                self.browser.find_elements(By.CLASS_NAME, 'artdeco-modal__confirm-dialog-btn')[0].click()
                time.sleep(random.uniform(3, 5))
                raise Exception("Failed to apply to job!")

        closed_notification = False
        time.sleep(random.uniform(3, 5))
        try:
            self.browser.find_element(By.CLASS_NAME, 'artdeco-modal__dismiss').click()
            closed_notification = True
        except:
            pass
        try:
            self.browser.find_element(By.CLASS_NAME, 'artdeco-toast-item__dismiss').click()
            closed_notification = True
        except:
            pass
        try:
            self.browser.find_element(By.CSS_SELECTOR, 'button[data-control-name="save_application_btn"]').click()
            closed_notification = True
        except:
            pass

        time.sleep(random.uniform(3, 5))

        if closed_notification is False:
            raise Exception("Could not close the applied confirmation window!")

        return True

    def home_address(self, form):
        print("Trying to fill up home address fields")
        # pdb.set_trace()  # Pause execution here for debugging

        try:
            groups = form.find_elements(By.CLASS_NAME, 'jobs-easy-apply-form-section__grouping')
            if len(groups) > 0:
                for group in groups:
                    lb = group.find_element(By.TAG_NAME, 'label').text.lower()
                    input_field = group.find_element(By.TAG_NAME, 'input')
                    if 'street' in lb:
                        self.enter_text(input_field, self.personal_info['Street address'])
                    elif 'city' in lb:
                        self.enter_text(input_field, self.personal_info['City'])
                        time.sleep(3)
                        input_field.send_keys(Keys.DOWN)
                        input_field.send_keys(Keys.RETURN)
                    elif 'zip' in lb or 'zip / postal code' in lb or 'postal' in lb:
                        self.enter_text(input_field, self.personal_info['Zip'])
                    elif 'state' in lb or 'province' in lb:
                        self.enter_text(input_field, self.personal_info['State'])
                    else:
                        pass
        except:
            pass

    def get_answer(self, question):
        if self.checkboxes[question]:
            return 'yes'
        else:
            return 'no'
        
    def additional_questions(self, form):
        print("Trying to fill up additional questions")

        questions = form.find_elements(By.CLASS_NAME, 'fb-dash-form-element')
        if not questions:
            print("No additional questions found in form")
            return

        # Dictionary mapping question types to handler functions
        question_handlers = {
            'radio': self._handle_radio_question,
            'text': self._handle_text_question,
            'date': self._handle_date_question,
            'dropdown': self._handle_dropdown_question,
            'checkbox': self._handle_checkbox_question
        }

        for question in questions:
            question_type, question_element = self._identify_question_type(question)
            if question_type is None:
                if self.debug:
                    print(f"Debug: Could not identify question type for element")
                continue

            try:
                handler = question_handlers.get(question_type)
                if handler:
                    handler(question_element)
                else:
                    if self.debug:
                        print(f"Debug: No handler defined for question type: {question_type}")
            except Exception as e:
                if self.debug:
                    print(f"Error processing {question_type} question: {str(e)}")

    def _identify_question_type(self, question):
        """Identify the type of question based on its structure."""
        try:
            if question.find_elements(By.TAG_NAME, 'fieldset'):
                return 'radio', question
            if question.find_elements(By.TAG_NAME, 'input') or question.find_elements(By.TAG_NAME, 'textarea'):
                return 'text', question
            if question.find_elements(By.CLASS_NAME, 'artdeco-datepicker__input'):
                return 'date', question
            if question.find_elements(By.TAG_NAME, 'select'):
                return 'dropdown', question
            if question.find_elements(By.XPATH, ".//label[input[@type='checkbox']]"):
                return 'checkbox', question
        except:
            pass
        return None, None

    def _handle_radio_question(self, question):
        """Handle radio button questions."""
        try:
            radio_fieldset = question.find_element(By.TAG_NAME, 'fieldset')
            question_span = radio_fieldset.find_element(By.CLASS_NAME, 'fb-dash-form-element__label').find_elements(By.TAG_NAME, 'span')[0]
            radio_text = question_span.text.lower()
            if self.debug:
                print(f"Radio question text: {radio_text}")

            radio_labels = radio_fieldset.find_elements(By.TAG_NAME, 'label')
            radio_options = [(i, text.text.lower()) for i, text in enumerate(radio_labels)]
            if self.debug:
                print(f"Radio options: {[opt[1] for opt in radio_options]}")

            if not radio_options:
                raise Exception("No radio options found")

            answer = self._get_radio_answer(radio_text, radio_options)
            to_select = None
            if answer:
                for i, radio in enumerate(radio_labels):
                    if answer in radio.text.lower():
                        to_select = radio_labels[i]
                        break
                if to_select is None and self.debug:
                    print(f"Debug: Answer '{answer}' not found in radio options")

            if to_select is None:
                if self.debug:
                    print(f"Debug: No predefined answer for radio question: {radio_text}")
                self.record_unprepared_question("radio", radio_text)
                ai_response = self.ai_response_generator.generate_response(
                    radio_text, response_type="choice", options=radio_options
                )
                to_select = radio_labels[ai_response] if ai_response is not None else radio_labels[-1]

            if self.debug:
                print(f"Selecting radio option: {to_select.text}")
            to_select.click()
        except Exception as e:
            if self.debug:
                print(f"Error in radio question handling: {str(e)}")
            raise

    def _get_radio_answer(self, radio_text, radio_options):
        """Determine the answer for a radio question."""
        if 'driver\'s licence' in radio_text or 'driver\'s license' in radio_text:
            return self.get_answer('driversLicence')
        elif any(keyword in radio_text.lower() for keyword in [
            'aboriginal', 'native', 'indigenous', 'tribe', 'first nations',
            'native american', 'native hawaiian', 'inuit', 'metis', 'maori',
            'aborigine', 'ancestral', 'native peoples', 'original people',
            'first people', 'gender', 'race', 'disability', 'latino', 'torres',
            'do you identify'
        ]):
            negative_keywords = ['prefer', 'decline', 'don\'t', 'specified', 'none', 'no']
            for option in radio_options:
                if any(neg_keyword in option[1].lower() for neg_keyword in negative_keywords):
                    return option[1]
        elif 'assessment' in radio_text:
            return self.get_answer("assessment")
        elif 'clearance' in radio_text:
            return self.get_answer("securityClearance")
        elif 'north korea' in radio_text:
            return 'no'
        elif 'previously employ' in radio_text or 'previous employ' in radio_text:
            return 'no'
        elif 'authorized' in radio_text or 'authorised' in radio_text or 'legally' in radio_text:
            return self.get_answer('legallyAuthorized')
        elif any(keyword in radio_text.lower() for keyword in [
            'certified', 'certificate', 'cpa', 'chartered accountant', 'qualification'
        ]):
            return self.get_answer('certifiedProfessional')
        elif 'urgent' in radio_text:
            return self.get_answer('urgentFill')
        elif 'commut' in radio_text or 'on-site' in radio_text or 'hybrid' in radio_text or 'onsite' in radio_text:
            return self.get_answer('commute')
        elif 'remote' in radio_text:
            return self.get_answer('remote')
        elif 'background check' in radio_text:
            return self.get_answer('backgroundCheck')
        elif 'drug test' in radio_text:
            return self.get_answer('drugTest')
        elif 'currently living' in radio_text or 'currently reside' in radio_text or 'right to live' in radio_text:
            return self.get_answer('residency')
        elif 'level of education' in radio_text:
            for degree in self.checkboxes['degreeCompleted']:
                if degree.lower() in radio_text:
                    return "yes"
        elif 'experience' in radio_text:
            if self.experience_default > 0:
                return 'yes'
            for experience in self.experience:
                if experience.lower() in radio_text:
                    return "yes"
        elif 'data retention' in radio_text:
            return 'no'
        elif 'sponsor' in radio_text:
            return self.get_answer('requireVisa')
        return None

    def _handle_text_question(self, question):
        """Handle text or numeric input questions."""
        try:
            question_text = question.find_element(By.TAG_NAME, 'label').text.lower()
            if self.debug:
                print(f"Text question text: {question_text}")

            try:
                txt_field = question.find_element(By.TAG_NAME, 'input')
            except:
                txt_field = question.find_element(By.TAG_NAME, 'textarea')

            # Check if field is pre-populated
            current_value = txt_field.get_attribute('value').strip()
            if current_value:
                if self.debug:
                    print(f"Field pre-populated with: {current_value}. Skipping.")
                return

            text_field_type = 'numeric' if 'numeric' in txt_field.get_attribute('id').lower() else 'text'
            to_enter = self._get_text_answer(question_text, text_field_type)

            if to_enter is None:
                if self.debug:
                    print(f"Debug: No predefined answer for text question: {question_text}")
                self.record_unprepared_question(text_field_type, question_text)
                response_type = "numeric" if text_field_type == 'numeric' else "text"
                to_enter = self.ai_response_generator.generate_response(
                    question_text, response_type=response_type
                )
                to_enter = to_enter if to_enter is not None else (0 if text_field_type == 'numeric' else " ‏‏‎ ")

            if self.debug:
                print(f"Entering text: {to_enter}")
            self.enter_text(txt_field, str(to_enter))
        except Exception as e:
            if self.debug:
                print(f"Error in text question handling: {str(e)}")
            raise

    def _get_text_answer(self, question_text, text_field_type):
        """Determine the answer for a text or numeric question."""
        if 'experience' in question_text or 'how many years in' in question_text or 'how many years' in question_text and text_field_type == 'numeric':
            for experience in self.experience:
                if experience.lower() in question_text:
                    return int(self.experience[experience])
            return None
        elif 'grade point average' in question_text:
            return self.university_gpa
        elif 'first name' in question_text and 'last name' not in question_text:
            return self.personal_info['First Name']
        elif 'last name' in question_text and 'first name' not in question_text:
            return self.personal_info['Last Name']
        elif 'location' in question_text:
            return 'Houston, Texas, United States'
        elif 'name' in question_text:
            return self.personal_info['First Name'] + " " + self.personal_info['Last Name']
        elif 'pronouns' in question_text:
            return self.personal_info['Pronouns']
        elif 'phone' in question_text:
            return self.personal_info['Mobile Phone Number']
        elif 'linkedin' in question_text:
            return self.personal_info['Linkedin']
        elif 'message to hiring' in question_text or 'cover letter' in question_text:
            return self.personal_info['MessageToManager']
        elif 'website' in question_text or 'github' in question_text or 'portfolio' in question_text:
            return self.personal_info['Website']
        elif 'notice' in question_text or 'weeks' in question_text:
            return int(self.notice_period) if text_field_type == 'numeric' else str(self.notice_period)
        elif 'salary' in question_text or 'expectation' in question_text or 'compensation' in question_text or 'CTC' in question_text:
            return int(self.salary_minimum) if text_field_type == 'numeric' else float(self.salary_minimum)
        return None

    def _handle_date_question(self, question):
        """Handle date picker questions."""
        try:
            date_picker = question.find_element(By.CLASS_NAME, 'artdeco-datepicker__input')
            date_picker.clear()
            date_picker.send_keys(date.today().strftime("%m/%d/%y"))
            time.sleep(3)
            date_picker.send_keys(Keys.RETURN)
            time.sleep(2)
            if self.debug:
                print("Filled date picker with today's date")
        except Exception as e:
            if self.debug:
                print(f"Error in date question handling: {str(e)}")
            raise

    def _handle_dropdown_question(self, question):
        """Handle dropdown questions."""
        try:
            question_text = question.find_element(By.TAG_NAME, 'label').text.lower()
            dropdown_field = question.find_element(By.TAG_NAME, 'select')
            if self.debug:
                print(f"Dropdown question text: {question_text}")

            select = Select(dropdown_field)
            options = [option.text for option in select.options]
            if self.debug:
                print(f"Dropdown options: {options}")

            choice = self._get_dropdown_answer(question_text, options)
            if choice is None:
                if self.debug:
                    print(f"Debug: No predefined answer for dropdown question: {question_text}")
                self.record_unprepared_question("dropdown", question_text)
                choices = [(i, option) for i, option in enumerate(options)]
                ai_response = self.ai_response_generator.generate_response(
                    question_text, response_type="choice", options=choices
                )
                choice = options[ai_response] if ai_response is not None else options[-1]

            if self.debug:
                print(f"Selecting dropdown option: {choice}")
            self.select_dropdown(dropdown_field, choice)
        except Exception as e:
            if self.debug:
                print(f"Error in dropdown question handling: {str(e)}")
            raise

    def _get_dropdown_answer(self, question_text, options):
        """Determine the answer for a dropdown question."""
        if 'proficiency' in question_text:
            for language in self.languages:
                if language.lower() in question_text:
                    return self.languages[language]
        elif 'clearance' in question_text or 'assessment' in question_text or 'commut' in question_text or 'on-site' in question_text or 'hybrid' in question_text or 'onsite' in question_text:
            answer = self.get_answer('securityClearance' if 'clearance' in question_text else 'assessment' if 'assessment' in question_text else 'commute')
            for option in options:
                if answer.lower() in option.lower():
                    return option
        elif 'country code' in question_text:
            return self.personal_info['Phone Country Code']
        elif 'north korea' in question_text or 'previously employed' in question_text or 'previous employment' in question_text:
            for option in options:
                if 'no' in option.lower():
                    return option
        elif 'sponsor' in question_text:
            answer = self.get_answer('requireVisa')
            for option in options:
                if answer.lower() in option.lower():
                    return option
        elif 'above 18' in question_text:
            for option in options:
                if 'yes' in option.lower():
                    return option
            return options[0]
        elif 'currently living' in question_text or 'currently reside' in question_text or 'authorized' in question_text or 'authorised' in question_text or 'citizenship' in question_text:
            answer = self.get_answer('residency' if 'reside' in question_text else 'legallyAuthorized')
            for option in options:
                if answer.lower() in option.lower() or ('no' in option.lower() and 'citizenship' in question_text and answer == 'yes'):
                    return option
        elif any(keyword in question_text.lower() for keyword in [
            'aboriginal', 'native', 'indigenous', 'tribe', 'first nations',
            'native american', 'native hawaiian', 'inuit', 'metis', 'maori',
            'aborigine', 'ancestral', 'native peoples', 'original people',
            'first people', 'gender', 'race', 'disability', 'latino'
        ]):
            negative_keywords = ['prefer', 'decline', 'don\'t', 'specified', 'none']
            for option in options:
                if any(neg_keyword in option.lower() for neg_keyword in negative_keywords):
                    return option
        elif 'experience' in question_text or 'understanding' in question_text or 'familiar' in question_text or 'comfortable' in question_text or 'able to' in question_text:
            answer = 'no'
            if self.experience_default > 0:
                answer = 'yes'
            else:
                for experience in self.experience:
                    if experience.lower() in question_text and self.experience[experience] > 0:
                        answer = 'yes'
                        break
            for option in options:
                if answer.lower() in option.lower():
                    return option
        return None

    def _handle_checkbox_question(self, question):
        """Handle checkbox questions (e.g., terms and service)."""
        try:
            clickable_checkbox = question.find_element(By.TAG_NAME, 'label')
            if self.debug:
                print(f"Clicking checkbox: {clickable_checkbox.text}")
            clickable_checkbox.click()
        except Exception as e:
            if self.debug:
                print(f"Error in checkbox question handling: {str(e)}")
            raise

    def unfollow(self):
        try:
            follow_checkbox = self.browser.find_element(By.XPATH,
                                                        "//label[contains(.,\'to stay up to date with their page.\')]").click()
            follow_checkbox.click()
        except:
            pass

    def send_resume(self):
        print("Trying to send resume")
        try:
            file_upload_elements = (By.CSS_SELECTOR, "input[name='file']")
            if len(self.browser.find_elements(file_upload_elements[0], file_upload_elements[1])) > 0:
                input_buttons = self.browser.find_elements(file_upload_elements[0], file_upload_elements[1])
                if len(input_buttons) == 0:
                    raise Exception("No input elements found in element")
                for upload_button in input_buttons:
                    upload_type = upload_button.find_element(By.XPATH, "..").find_element(By.XPATH,
                                                                                          "preceding-sibling::*")
                    if 'resume' in upload_type.text.lower():
                        upload_button.send_keys(self.resume_dir)
                    elif 'cover' in upload_type.text.lower():
                        if self.cover_letter_dir != '':
                            upload_button.send_keys(self.cover_letter_dir)
                        elif 'required' in upload_type.text.lower():
                            upload_button.send_keys(self.resume_dir)
        except:
            print("Failed to upload resume or cover letter!")
            pass

    def enter_text(self, element, text):
        element.clear()
        element.send_keys(text)

    def select_dropdown(self, element, text):
        select = Select(element)
        select.select_by_visible_text(text)

    # Radio Select
    def radio_select(self, element, label_text, clickLast=False):
        label = element.find_element(By.TAG_NAME, 'label')
        if label_text in label.text.lower() or clickLast == True:
            label.click()

    # Contact info fill-up
    def contact_info(self, form):
        print("Trying to fill up contact info fields")
        try:
            # Find all form groups within the contact info form
            form_groups = form.find_elements(By.CLASS_NAME, 'form-group')
            for group in form_groups:
                label = group.find_element(By.TAG_NAME, 'label').text.lower()
                if 'first name' in label:
                    try:
                        first_name_field = group.find_element(By.ID, 'first-name')
                        self.enter_text(first_name_field, self.personal_info['First Name'])
                        if self.debug:
                            print(f"Filled First Name with: {self.personal_info['First Name']}")
                    except Exception as e:
                        print(f"Could not fill First Name field: {str(e)}")
                elif 'last name' in label:
                    try:
                        last_name_field = group.find_element(By.ID, 'last-name')
                        self.enter_text(last_name_field, self.personal_info['Last Name'])
                        if self.debug:
                            print(f"Filled Last Name with: {self.personal_info['Last Name']}")
                    except Exception as e:
                        print(f"Could not fill Last Name field: {str(e)}")
                elif 'phone number' in label or 'country code' in label:
                    try:
                        country_code_picker = group.find_element(By.ID, 'country-code')
                        self.select_dropdown(country_code_picker, self.personal_info['Phone Country Code'])
                        if self.debug:
                            print(f"Selected Phone Country Code: {self.personal_info['Phone Country Code']}")
                    except Exception as e:
                        print(f"Could not select Phone Country Code: {str(e)}")
                    try:
                        phone_number_field = group.find_element(By.XPATH, '//input[contains(@id,"phoneNumber")][contains(@id,"nationalNumber")]')
                        self.enter_text(phone_number_field, self.personal_info['Mobile Phone Number'])
                        if self.debug:
                            print(f"Filled Phone Number with: {self.personal_info['Mobile Phone Number']}")
                    except Exception as e:
                        print(f"Could not enter Phone Number: {str(e)}")
        except Exception as e:
            print(f"Error processing contact info form: {str(e)}")
    def fill_up(self):
        try:
            easy_apply_modal_content = self.browser.find_element(By.CLASS_NAME, "jobs-easy-apply-modal__content")
            form = easy_apply_modal_content.find_element(By.TAG_NAME, 'form')
            try:
                label = form.find_element(By.TAG_NAME, 'h3').text.lower()
                if 'home address' in label:
                    self.home_address(form)
                elif 'contact info' in label:
                    self.contact_info(form)
                elif 'resume' in label:
                    self.send_resume()
                else:
                    self.additional_questions(form)
            except Exception as e:
                print("An exception occurred while filling up the form:")
                print(e)
        except:
            print("An exception occurred while searching for form in modal")

    def write_to_file(self, company, job_title, link, location, search_location):
        to_write = [company, job_title, link, location, search_location, datetime.now()]
        file_path = self.file_name + ".csv"
        print(f'updated {file_path}.')

        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(to_write)

    def record_unprepared_question(self, answer_type, question_text):
        to_write = [answer_type, question_text]
        file_path = self.unprepared_questions_file_name + ".csv"

        try:
            with open(file_path, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(to_write)
                print(f'Updated {file_path} with {to_write}.')
        except:
            print(
                "Special characters in questions are not allowed. Failed to update unprepared questions log.")
            print(question_text)

    def scroll_slow(self, scrollable_element, start=0, end=3600, step=100, reverse=False):
        if reverse:
            start, end = end, start
            step = -step

        for i in range(start, end, step):
            self.browser.execute_script("arguments[0].scrollTo(0, {})".format(i), scrollable_element)
            time.sleep(random.uniform(0.1, .6))

    def avoid_lock(self):
        if self.disable_lock:
            return

        pyautogui.keyDown('ctrl')
        pyautogui.press('esc')
        pyautogui.keyUp('ctrl')
        time.sleep(1.0)
        pyautogui.press('esc')

    def get_base_search_url(self, parameters):
        remote_url = ""
        lessthanTenApplicants_url = ""
        newestPostingsFirst_url = ""

        if parameters.get('remote'):
            remote_url = "&f_WT=2"
        else:
            remote_url = ""
            # TO DO: Others &f_WT= options { WT=1 onsite, WT=2 remote, WT=3 hybrid, f_WT=1%2C2%2C3 }

        if parameters['lessthanTenApplicants']:
            lessthanTenApplicants_url = "&f_EA=true"

        if parameters['newestPostingsFirst']:
            newestPostingsFirst_url += "&sortBy=DD"

        level = 1
        experience_level = parameters.get('experienceLevel', [])
        experience_url = "f_E="
        for key in experience_level.keys():
            if experience_level[key]:
                experience_url += "%2C" + str(level)
            level += 1

        distance_url = "?distance=" + str(parameters['distance'])

        job_types_url = "f_JT="
        job_types = parameters.get('jobTypes', [])
        # job_types = parameters.get('experienceLevel', [])
        for key in job_types:
            if job_types[key]:
                job_types_url += "%2C" + key[0].upper()

        date_url = ""
        dates = {"all time": "", "month": "&f_TPR=r2592000", "week": "&f_TPR=r604800", "24 hours": "&f_TPR=r86400"}
        date_table = parameters.get('date', [])
        for key in date_table.keys():
            if date_table[key]:
                date_url = dates[key]
                break

        easy_apply_url = "&f_AL=true"

        extra_search_terms = [distance_url, remote_url, lessthanTenApplicants_url, newestPostingsFirst_url, job_types_url, experience_url]
        extra_search_terms_str = '&'.join(
            term for term in extra_search_terms if len(term) > 0) + easy_apply_url + date_url

        return extra_search_terms_str

    def next_job_page(self, position, location, job_page):
        self.browser.get("https://www.linkedin.com/jobs/search/" + self.base_search_url +
                         "&keywords=" + position + location + "&start=" + str(job_page * 25))

        self.avoid_lock()