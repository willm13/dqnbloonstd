from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC
import time
import numpy as np
import cv2
import pytesseract
from PIL import Image
import io
import random

TOWER_TYPE = [[0, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]]

""" PLACEMENT_LOCATIONS = {1: (597, 322), 2: (660, 322), 3: (725, 322), 4: (597, 288), 5: (660, 288), 6: (725, 288), 7: (597, 255), 8: (660, 255), 9: (725, 255),
                       10: (597, 205), 11: (637, 205), 12: (597, 160), 13: (637, 160), 14: (597, 124), 15: (637, 124),
                       16: (723, 153), 17: (779, 153), 18: (829, 153), 19: (877, 153), 20: (929, 153), 21: (979, 153), 22: (1024, 153),
                       23: (829, 187), 24: (877, 187), 25: (929, 184), 26: (979, 184), 27: (1024, 184), 
                       28: (829, 223), 29: (877, 223),
                       30: (829, 260), 31: (877, 260),
                       32: (829, 300), 33: (877, 300),
                       34: (829, 335), 35: (877, 335),
                       36: (829, 375), 37: (877, 375), 38: (935, 375), 39: (991, 375), 40: (1037, 375),
                       41: (1037, 413), 42: (991, 413), 43: (935, 413), 44: (877, 413), 45: (829, 413), 46: (779, 413), 47: (722, 413), 48: (666, 413), 49: (591, 414),
                       50: (593, 466), 51: (659, 466), 52: (593, 520), 53: (659, 520), 54: (593, 576), 55: (659, 576),
                       56: (746, 503), 57: (802, 503), 58: (746, 540), 59: (802, 540), 60: (746, 577), 61: (802, 577),
                       62: (896, 558), 63: (949, 558), 64: (897, 598), 65: (949, 598), 66: (896, 646), 67: (949, 646),
                       68: (1057, 504), 69: (1055, 545), 70: (1057, 589), 71: (1057, 630),
                       72: (969, 260), 73: (1026, 260), 74: (1073, 260)} """

PLACEMENT_LOCATIONS = {0: (597, 322), 1: (660, 322), 2: (725, 322), 3: (597, 288), 4: (660, 288), 5: (725, 288), 6: (597, 255), 7: (660, 255), 8: (725, 255), 9: (597, 205), 10: (637, 205), 11: (597, 160), 12: (637, 160), 13: (597, 124), 14: (637, 124), 15: (723, 153), 16: (779, 153), 17: (829, 153), 18: (877, 153), 19: (929, 153), 20: (979, 153), 21: (1024, 153), 22: (829, 187), 23: (877, 187), 24: (929, 184), 25: (979, 184), 26: (1024, 184), 27: (829, 223), 28: (877, 223), 29: (829, 260), 30: (877, 260), 31: (829, 300), 32: (877, 300), 33: (829, 335), 34: (877, 335), 35: (829, 375), 36: (877, 375), 37: (935, 375), 38: (991, 375), 39: (1037, 375), 40: (1037, 413), 41: (991, 413), 42: (935, 413), 43: (877, 413), 44: (829, 413), 45: (779, 413), 46: (722, 413), 47: (666, 413), 48: (591, 414), 49: (593, 466), 50: (659, 466), 51: (593, 520), 52: (659, 520), 53: (593, 576), 54: (659, 576), 55: (746, 503), 56: (802, 503), 57: (746, 540), 58: (802, 540), 59: (746, 577), 60: (802, 577), 61: (896, 558), 62: (949, 558), 63: (897, 598), 64: (949, 598), 65: (896, 646), 66: (949, 646), 67: (1057, 504), 68: (1055, 545), 69: (1057, 589), 70: (1057, 630), 71: (969, 260), 72: (1026, 260), 73: (1073, 260)}

# MONKEY INFO
M_X = 1191 
M_Y = 205
M_C = 250
M_C_U1 = 180
M_C_U2 = 80

T_X = 1242
T_Y = 202
T_C = 350
T_C_U1 = 180
T_C_U2 = 90

# Ice Info
I_X = 1287
I_Y = 202
I_C = 385
I_C_U1 = 270
I_C_U2 = 180

# Cannon Info
C_X = 1336
C_Y = 202
C_C = 520
C_C_U1 = 200
C_C_U2 = 250

# Boomerang
B_X = 1287
B_Y = 246
B_C  = 475
B_C_U1 = 190
B_C_U2 = 230

# Super Monkey  
S_X = 1336
S_Y = 246
S_C = 3600
S_C_U1 = 9999
S_C_U2 = 9999

class Browser:


    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        ad_block = r'C:\4366\project\1.56.0_0'
        chrome_options = Options()
        chrome_options.add_argument('load-extension=' + ad_block)
        service = Service(executable_path='chromedriver.exe')

        self.board_state = [0] * (592) # 74 * 6 tower types, 74 x 2 upgrades
        self.money = 0
        self.lives = 100
        self.last_lives = 100
        self.rounds = 1
        self.last_round = 1
        self.url = "https://www.crazygames.com/game/bloons-tower-defense-2"
        
        self.driver = webdriver.Chrome(service = service, options=chrome_options)
        self.driver.delete_all_cookies()

    def launch(self):
        self.driver.get(self.url)
        time.sleep(1)
        self.driver.maximize_window()   
    
    def cleanup(self):
        self.driver.close()

    def click(self, x, y):
        action_chains = ActionChains(self.driver)
        action_chains.move_by_offset(x, y)
        action_chains.click()
        action_chains.perform()
        action_chains.reset_actions()
    
    def invalid_placement(self, img, rgb_value, tolerance=10):
        # Convert the input RGB value to BGR color format
        bgr_value = (rgb_value[2], rgb_value[1], rgb_value[0])
        
        # Calculate the lower and upper bounds for each color component
        lower_bound = np.array(bgr_value) - tolerance
        upper_bound = np.array(bgr_value) + tolerance
        
        # Test if any pixels in the image fall within the tolerance range
        return np.any(np.all((img >= lower_bound) & (img <= upper_bound), axis=2))
    
    def add_tower(self, placement_location, tower, x, y):
        if tower == 'monkey':
            tower = 0
            self.click(M_X, M_Y)
        elif tower == 'tack':
            tower = 1
            self.click(T_X, T_Y)
        elif tower == 'freeze':
            tower = 2
            self.click(I_X, I_Y)
        elif tower == 'cannon':
            tower = 3
            self.click(C_X, C_Y)
        elif tower == 'boom':
            tower = 4
            self.click(B_X, B_Y)
        elif tower == 'super':
            tower = 5
            self.click(S_X, S_Y)
        
        x_copy = x
        y_copy = y
        attempts = 0
        # hover so we can see if valid placement

        while True:
            action_chains = ActionChains(self.driver)
            action_chains.move_by_offset(x_copy, y_copy)
            action_chains.perform()
            screen = self.driver.get_screenshot_as_png()
            action_chains.reset_actions()

            # screen capture of area around monkey
            img = np.asarray(bytearray(screen), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)

            x_min = max(0, x_copy - 100)
            x_max = min(img.shape[1], x_copy + 100)
            y_min = max(0, y_copy - 100)
            y_max = min(img.shape[0], y_copy + 100)
            cropped_img = img[y_min:y_max, x_min:x_max]
            #print(cropped_img)
            cv2.imwrite("aaa.png", cropped_img)
            if self.invalid_placement(cropped_img, (207, 130, 153)) == True:
                x_copy = random.randint(x - 10, x + 10)
                y_copy = random.randint(y - 10, y + 10)
                attempts += 1
                if attempts > 5:
                    return -1
            else:
                break
                
        # place 
        self.click(x_copy, y_copy)
        self.board_state[(placement_location) * 6 + tower] = 1
        #print(self.board_state)

    def upgrade_tower(self, tower_num, upgrade_type):
        location = PLACEMENT_LOCATIONS.get(tower_num)
        self.click(location[0], location[1])
        if upgrade_type == 0:
            self.click(1220, 408)
        else:
            self.click(1320, 408)
        self.board_state[444 + (tower_num)*2 + upgrade_type] = 1


    def start_round(self):
        start_x, start_y = 1265, 630
        self.click(start_x, start_y)
    
    def reset_game(self):
        self.board_state = [0] * (592)
        self.money = 0
        self.lives = 100
        self.last_lives = 100
        self.rounds = 1
        self.last_round = 1
        end_x, end_y = 1320, 670
        self.click(end_x, end_y)
        select_easy_x, select_easy_y = 715, 180
        self.click(select_easy_x, select_easy_y)
        
    def start_game(self):
        iframe = WebDriverWait(self.driver, 5).until(EC.presence_of_element_located((By.CSS_SELECTOR, "iframe")))
        self.driver.switch_to.frame(iframe)

        # Press play
        play_button = self.driver.find_element(By.CLASS_NAME, 'css-g39b1')
        play_button.click()
        self.driver.switch_to.default_content()
        #self.driver.save_screenshot("x.png")

        # Cutscene
        time.sleep(20)

        #Select mode
        select_easy_x, select_easy_y = 715, 180
        self.click(select_easy_x, select_easy_y)

        # Start game
        self.start_round()

    def get_game_state(self):
        self.start_round()
        screen = self.driver.get_screenshot_as_png()
        img = np.asarray(bytearray(screen), dtype=np.uint8)

        img = cv2.imdecode(img, cv2.IMREAD_COLOR) #make it usable for cv2
        img = img[68:682, 400:1527] #crop to just the game

        # money
        x, y, w, h = 900, 14, 70, 32
        roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (h, w) = gray_roi.shape[:2]
        gray_roi = cv2.resize(gray_roi, (w*4, h*4))
        gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        money = pytesseract.image_to_string(gray_roi, config="digits")
        cv2.imwrite("processed_money.png", gray_roi)

        if money != '':
            try:
                self.money = int(money)
            except ValueError:
                pass
        print(f"Current Money: {self.money}")

        #cv2.imshow("ROI", gray_roi)
        #cv2.waitKey()
        #cv2.destroyAllWindows()

        #lives
        x, y, w, h = 900, 48, 70, 32
        roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (h, w) = gray_roi.shape[:2]
        gray_roi = cv2.resize(gray_roi, (w*3, h*3))
        gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        lives = pytesseract.image_to_string(gray_roi, config="digits")
        try:
            self.lives = int(lives)
        except ValueError:
            pass
        print(f"Current lives: {self.lives}")

        #rounds
        x, y, w, h = 160, 572, 70, 52
        roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        (h, w) = gray_roi.shape[:2]
        gray_roi = cv2.resize(gray_roi, (w*3, h*3))
        gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rounds = pytesseract.image_to_string(gray_roi, config="--psm 6 digits")
        try:
            self.rounds = int(rounds)
        except ValueError:
            pass
        print(f"Last round: {self.last_round}, Current round: {self.rounds}")

        print(f"Current State: {[self.money, self.lives, self.rounds] + self.board_state}")

        return [self.money, self.lives, self.rounds] + self.board_state
            
    def tower_type(self, num):
        if num == 0:
            return "monkey"
        elif num == 1:
            return "tack"
        elif num == 2:
            return "freeze"
        elif num == 3:
            return "cannon"
        elif num == 4:
            return "boom"
        elif num == 5:
            return "super"
    
    def can_afford_place(self, num):
        if num == 0:
            return self.money >= M_C
        elif num == 1:
            return self.money >= T_C
        elif num == 2:
            return self.money >= I_C
        elif num == 3:
            return self.money >= C_C
        elif num == 4:
            return self.money >= B_C
        elif num == 5:
            return self.money >= S_C
        
        return False
    
    def can_afford_upgrade(self, num, upgrade_type):
        if num == 0:
            if upgrade_type == 0:
                return self.money >= M_C_U1
            else:
                return self.money >= M_C_U1
        elif num == 1:
            if upgrade_type == 0:
                return self.money >= T_C_U1
            else:
                return self.money >= T_C_U1
        elif num == 2:
            if upgrade_type == 0:
                return self.money >= I_C_U1
            else:
                return self.money >= I_C_U1
        elif num == 3:
            if upgrade_type == 0:
                return self.money >= C_C_U1
            else:
                return self.money >= C_C_U1
        elif num == 4:
            if upgrade_type == 0:
                return self.money >= B_C_U1
            else:
                return self.money >= B_C_U1
        elif num == 5:
            if upgrade_type == 0:
                return self.money >= S_C_U1
            else:
                return self.money >= S_C_U1
        
        return

    def valid_action(self, action):

        if action <= 443:
            placement_location = (action // 6)
            tower_type = action % 6
            for i in range(6):
                if self.board_state[placement_location*6 + i] == 1:
                    #print("Tower already exists in this location")
                    return False #INVALID PLACING
            if self.can_afford_place(tower_type):
                #print(f"Can afford at {placement_location} type {self.tower_type(tower_type)}")
                return True
            else:
                #print(f"Could not afford placing {self.tower_type(tower_type)}")
                return False
        elif action >= 444 and action < 592:
            upgrade_location = (action - 444) // 2
            upgrade_type = action % 2
            for i in range(6):
                #print(upgrade_location*6 + i)
                if self.board_state[upgrade_location*6 + i] == 1:
                    #print("HELLO!!!!")
                    break
                elif i == 5:
                    print(upgrade_location*6 + i)

                    #print("Could not upgrade due to invaid tower")
                    return False
            if self.board_state[action] == 1:
                    #print("That position and upgrade is already taken")
                    return False # already upgraded
            elif self.can_afford_upgrade(i, upgrade_type):
                    #print("Upgrade can be afforded")
                    return True
            else:
                return False
            
        return True
    
    def get_reward(self, action):
        LOSS = -50
        ROUND_PASSED_BASE = 2
        DO_NOTHING = -0.5
        LIVE_LOSS = -0.5

        total_reward = 0
        if self.lives < self.last_lives:
            total_reward += LIVE_LOSS
            self.last_lives = self.lives
        if action == 592:
            total_reward += DO_NOTHING
        if self.rounds > self.last_round:
            print("-Round Passed, rewarded-")
            self.last_round = self.rounds
            total_reward += (ROUND_PASSED_BASE * (1.05 ** self.rounds))
        if self.lives <= 50:
            total_reward += LOSS
        print(f"--Step Reward: {round(total_reward, 2)}--")
        return round(total_reward, 2)


    def step(self, action):
        if action <= 443:
            placement_location = (action) // 6
            tower_type_index = action % 6
            coords = PLACEMENT_LOCATIONS.get(placement_location)
            #print(coords)
            #print(type(coords))
            self.add_tower(placement_location, self.tower_type(tower_type_index), coords[0], coords[1])
        elif action >= 444 and action < 592:
            upgrade_location = (action - 444) // 2
            upgrade_type = action % 2
            #print(f"Upgrade tower at position: {upgrade_location} type: {upgrade_type}")
            self.upgrade_tower(upgrade_location, upgrade_type)
        else:
            pass # do nothing action

        reward = self.get_reward(action)
        done = True if self.lives <= 50 else False
        print(f"Terminal State?: {done}")
        # wait 5 seconds before returning next state
        time.sleep(5)

        while True:
            #print("WE ARE STUCK HERE")
            next_state = self.get_game_state()
            if next_state is not None:
                #print("IT AINT BREAKING")
                break
        #print("SOMEHOW THIS IS NULL")

        return next_state, reward, done






    
    



if __name__ == '__main__':
    browser = Browser()
    browser.launch()
    browser.start_game()
    browser.valid_action(18)
    browser.step(18)
    browser.valid_action(450)
    browser.step(450)
    time.sleep(1)
    browser.reset_game()
    browser.valid_action(438)
    browser.step(438)
    browser.valid_action(590)
    browser.step(590)
    browser.valid_action(590)

    time.sleep(20)