use std::process::Command;
use std::str;
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::Duration;

use std::fs::File;
use std::io;
use std::path::Path;

use reqwest;

use enigo::*;
use rand::Rng;
use rocket::State;

use crate::{TradeBotInfo, database_functions, ReadyState};
use crate::TradersContainer;

use crate::enigo_functions;

// Opens the windows search bar and searches for a given title and opens it
fn start_game(enigo: &mut Enigo, launcher_name: &str) {
    enigo.key_click(Key::Meta);
    sleep(Duration::from_millis(1000));
    enigo.key_sequence_parse(launcher_name);
    sleep(Duration::from_millis(2000));
    enigo.key_click(Key::Return);
}

// This function does the following:
// 1. Opens the blacksmith launcher and presses play
// 2. Goes into the lobby.
// 3. Changes the TradeBotInfo ready variable to true when ready.
pub async fn open_game_go_to_lobby(bot_info: Arc<Mutex<TradeBotInfo>>) {
    let enigo = Arc::new(Mutex::new(Enigo::new()));

    println!("Opening game!");
    {
        let mut bot_info = bot_info.lock().unwrap();
        bot_info.ready = ReadyState::Starting;
    }
    //tokio::time::sleep(tokio::time::Duration::from_secs(10000)).await;

    let mut enigo = enigo.lock().unwrap();

    // Minimizes all tabs so that only the game is opened. To avoid clicking on other tabs
    enigo.key_sequence_parse("{+META}m{-META}");

    // Start the launcher
    start_game(&mut enigo, "blacksmith");

    // Run the launcher play button detector
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/play.png")
        .output()
        .expect("Failed to execute command");

    println!("Output: {:?}", output);

    match enigo_functions::click_buton(&mut enigo, output, false, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // Now we are opening the game
    // Run the "Ok" button detector (Will run once we enter the game)
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/okay_start.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // Run the "Enter the lobby" button detector
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/enter_lobby.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // Now the bot is in the lobby "play" tab
    let mut info = bot_info.lock().unwrap();
    info.ready = ReadyState::True;
    info.id = String::from("Middleman2");
}

// It waits untill a trade request is sent by the discord bot
pub fn collect_gold_fee(
    enigo: &State<Arc<Mutex<Enigo>>>,
    bot_info: &State<Arc<Mutex<TradeBotInfo>>>,
    in_game_id: &str,
    traders_container: &State<Arc<Mutex<TradersContainer>>>,
) {
    let mut enigo = enigo.lock().unwrap();

    let info = bot_info.lock().unwrap();

    // If the bot is not ready then it will run the open game function
    // If the bot is starting then it will wait for the bot to be ready
    // If the bot is ready then it will continue as normal
    'wait_loop: loop{
        let bot_info_clone = bot_info.inner().clone();
        match info.ready {
            ReadyState::False => {
                tokio::spawn(async move {
                    open_game_go_to_lobby(bot_info_clone).await;
                });
            },
            ReadyState::Starting => sleep(Duration::from_secs(2)),
            ReadyState::True => break 'wait_loop,
        }
    }
    // Goes into the trading tab and connects to bards trade post.
    // Why bard? Because it has the least amount of active traders and therefore not as demanding to be in.
    // Run the "Trade" tab detector
    send_trade_request(in_game_id);

    // Check if user has put in 50 gold for the trade fee
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/gold_fee2.png")
        .arg("S")
        .output();

    match output {
        Ok(_) => println!("User put in the gold fee."),
        Err(_) => {
            println!("User did not put in gold fee..");
            return_to_lobby();
            return;
        }
    }

    // Click the checkbox
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/trade_checkbox.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // Double check that the total gold is still the same in the trade confirmation window
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/gold_fee_double_check.png")
        .arg("S")
        .output();

    match output {
        Ok(_) => println!("User put in the gold fee."),
        Err(_) => {
            println!("User did not put in gold fee..");
            return_to_lobby();
            return;
        }
    }

    // Click the magnifying glasses on top of the items
    let output = Command::new("python")
        .arg("python_helpers/inspect_items.py")
        .output()
        .expect("Failed to execute command");

    // Convert the output bytes to a string
    let output_str = str::from_utf8(&output.stdout).unwrap().trim();

    // Split the string on newlines to get the list of coordinates
    let coords: Vec<&str> = output_str.split('\n').collect();

    // Now, coords contains each of the coordinates
    for coord_str in coords.iter() {
        let coord: Vec<i32> = coord_str
            .split_whitespace()
            .map(|s| s.parse().expect("Failed to parse coordinate"))
            .collect();

        if coord.len() == 4 {
            let (x1, y1, x2, y2) = (coord[0], coord[1], coord[2], coord[3]);

            let mut rng = rand::thread_rng();

            // Salt the pixels so that it does not click the same pixel every time.
            let salt = rng.gen_range(-9..9);

            // Gets the middle of the detected play button and clicks it
            let middle_point_x = ((x2 - x1) / 2) + x1 + salt;
            let middle_point_y = ((y2 - y1) / 2) + y1 + salt;

            match enigo_functions::click_buton_right_direct(
                &mut enigo,
                middle_point_x,
                middle_point_y,
                true,
                true,
                0,
                0,
            ) {
                Ok(_) => println!("Successfully clicked button!"),
                Err(err) => println!("Got error while trying to click button: {:?}", err),
            }
        }
    }

    // Click the final checkpoint to get the 50 gold fee
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/trade_checkbox.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // When paid, set has_paid_gold_fee to true
    let mut traders = traders_container.lock().unwrap();
    let trader = traders.get_trader_by_in_game_id(in_game_id);

    // Check if trader exists
    match trader {
        Some(trader) => {
            match database_functions::set_gold_fee_status(trader.discord_channel_id.as_str(), trader.discord_id.as_str(), true) {
                Ok(_) => println!("Succesfully updated gold fee status!"),
                Err(err) => println!("Could not update gold status: Error \n{}", err),
            }
        },
        None => println!("Trader not found"),
    }

    // Make a copy of trader discord id. Else it would use traders as both mutable and imutable.
    let trader_discord_id = trader.unwrap().discord_id.as_str();
    let trader_discord_id_copy: String = String::from(trader_discord_id);
    traders.update_gold_fee_status(trader_discord_id_copy.as_str(), true);
}


pub fn collect_items( 
    enigo: &State<Arc<Mutex<Enigo>>>,
    bot_info: &State<Arc<Mutex<TradeBotInfo>>>,
    in_game_id: &str,
    traders_container: &State<Arc<Mutex<TradersContainer>>>,
) {
    let mut enigo = enigo.lock().unwrap();

    let info = bot_info.lock().unwrap();

    // If the bot is not ready then it will run the open game function
    // If the bot is starting then it will wait for the bot to be ready
    // If the bot is ready then it will continue as normal
    'wait_loop: loop{
        let bot_info_clone = bot_info.inner().clone();
        match info.ready {
            ReadyState::False => {
                tokio::spawn(async move {
                    open_game_go_to_lobby(bot_info_clone).await;
                });
            },
            ReadyState::Starting => sleep(Duration::from_secs(2)),
            ReadyState::True => break 'wait_loop,
        }
    }
        
        // Get the trader with that in-game name
    let traders = traders_container.lock().unwrap();
    let trader = traders.get_trader_by_in_game_id(in_game_id);

    // Get channel and discord id
    let channel_id = trader.unwrap().discord_channel_id.as_str();
    let discord_id = trader.unwrap().discord_id.as_str();

    let has_paid_fee = database_functions::has_paid_fee(channel_id, discord_id).unwrap();

    if !has_paid_fee {
        return;
    }

    // Go into the trading tab and send a trade to the trader. Exact same as before with the gold fee.
    send_trade_request(trader.unwrap().in_game_id.as_str());


    // Now we are in the trading window with the trader

    // Loop through the items in the trader struct for this trader and use obj detection to check if the item is present
    // If item is present then add it to list. Once it cannot find any more items in the trading window (Wait at least 30 seconds after detection an item so that the trader has time to put in the stuff) it should accept the items

    // Download 1 image set into temp_images folder at a time and check for a match
    let info_vec = &trader.unwrap().info_images;
    let item_vec = &trader.unwrap().item_images;

    // For each image pair. Download the pair and if there is a matching pair in the trading window, add it to list in memory.
    // After trading successfully and double checking in inspect window, change status to "in escrow" for the traded items in the database.
    let mut trading_window_items = Vec::new();

    for item in item_vec.iter(){
        match download_image(&item, "temp_images/item/image.png") {
            Ok(_) => println!("Successfully downloaded item image"),
            Err(err) => {
                println!("Could not download image. Error \n{}", err);
                return;
            }
        }

        let output = Command::new("python")
            .arg("python_helpers/multi_obj_detection.py")
            .arg("temp_images/item/image.png")
            .output()
            .expect("Failed to execute command");

        // Convert the output bytes to a string
        let output_str = str::from_utf8(&output.stdout).unwrap().trim();

        // Split the string on newlines to get the list of coordinates
        let coords: Vec<&str> = output_str.split('\n').collect();

        // Now, coords contains each of the coordinates
        for coord_str in coords.iter() {
            let coord: Vec<i32> = coord_str
                .split_whitespace()
                .map(|s| s.parse().expect("Failed to parse coordinate"))
                .collect();

            if coord.len() == 4 {
                let (x1, y1, x2, y2) = (coord[0], coord[1], coord[2], coord[3]);

                let mut rng = rand::thread_rng();

                // Salt the pixels so that it does not click the same pixel every time.
                let salt = rng.gen_range(-9..9);

                // Gets the middle of the detected play button and clicks it
                let middle_point_x = ((x2 - x1) / 2) + x1 + salt;
                let middle_point_y = ((y2 - y1) / 2) + y1 + salt;

                match enigo_functions::move_to_location_fast(
                    &mut enigo,
                    middle_point_x,
                    middle_point_y,
                ) {
                    Ok(_) => println!("Successfully moved to this location!"),
                    Err(err) => println!("Got error while trying to move cursor: {:?}", err),
                }

                // Tries to match every info image with the item and if there is a match then it will add it to the temporary vector variable.
                for info_image in info_vec.iter() {
                    match download_image(info_image, "temp_images/info/image.png") {
                        Ok(_) => println!("Successfully downloaded info image"),
                        Err(err) => {
                            println!("Could not download image. Error \n{}", err);
                            return;
                        }
                    }

                    // SHOULD USE A VERSION OF OBJ DETECTION WITH A FASTER TIMEOUT. So that it wont wait for 4 minutes of there is no match
                    let output = Command::new("python")
                        .arg("python_helpers/obj_detection.py")
                        .arg("temp_images/info/item.png")
                        .output();
    
                    match output {
                        Ok(_) => {
                            println!("Found match!");
                            trading_window_items.push((info_image, item));
                        },
                        Err(_) => println!("No match. Checking next..."),
                    }
                } 
            }
        }
    }
    // Accept trade
}



fn send_trade_request(in_game_id: &str) {
    let mut enigo = Enigo::new();


    // Goes into the trading tab and connects to bards trade post.
    // Why bard? Because it has the least amount of active traders and therefore not as demanding to be in.
    // Run the "Trade" tab detector
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/trade_tab.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // Now enter bards trading post
    // Run the "bard_trade" button detector
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/bard_trade.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    //It now sends a trade to the player
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/find_id.png")
        .output()
        .expect("Failed to execute command");

    // Search after the trader in the trade tab
    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    let user_is_in_trade: bool;

    // Type in the name of the trader
    let in_game_id_lower = in_game_id.to_lowercase();
    let in_game_id_lower_str_red: &str = &in_game_id_lower;
    enigo.key_sequence_parse(in_game_id_lower_str_red);

    // This runs the obj_detection script which tries to find the trade button.
    // If the person is not in the game, then there will be no trade button to press.
    // The obj_detection script runs for 4 minutes

    // Clicks directly on the first person below the bot, which should be the player to trade with.
    match enigo_functions::click_buton_right_direct(&mut enigo, 1824, 312, true, false, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }

    // Send a trade request
    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/trade_send_request.png")
        .output();

    user_is_in_trade = match &output {
        Ok(_) => true,
        Err(_) => false,
    };
    if user_is_in_trade {
        match enigo_functions::click_buton(&mut enigo, output.unwrap(), true, 0, 0) {
            Ok(_) => println!("Successfully clicked button!"),
            Err(err) => println!("Got error while trying to click button: {:?}", err),
        }
    }
    // Else go back to main window and return.
    else {
        return_to_lobby();
        return;
    }
}

fn return_to_lobby() {
    let mut enigo = Enigo::new();

    let output = Command::new("python")
        .arg("python_helpers/obj_detection.py")
        .arg("images/play_tab.png")
        .output()
        .expect("Failed to execute command");

    match enigo_functions::click_buton(&mut enigo, output, true, 0, 0) {
        Ok(_) => println!("Successfully clicked button!"),
        Err(err) => println!("Got error while trying to click button: {:?}", err),
    }
    return;
}


fn download_image(url: &str, save_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure the 'temp_images' directory exists
    if !Path::new("temp_images").exists() {
        std::fs::create_dir("temp_images")?;
    }

    // Perform a blocking HTTP GET request
    let response = reqwest::blocking::get(url)?;
    
    // Ensure the request was successful
    if response.status().is_success() {
        // Open a file to write the image data
        let mut file = File::create(save_path)?;

        // Copy the response data to the file
        let response_body = response.bytes()?;
        io::copy(&mut response_body.as_ref(), &mut file)?;

        println!("Image downloaded to '{}'", save_path);
    } else {
        return Err(Box::new(io::Error::new(io::ErrorKind::Other, "Failed to download image")));
    }

    Ok(())
}