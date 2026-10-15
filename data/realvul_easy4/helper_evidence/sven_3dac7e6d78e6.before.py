#!/bin/python3
import discord
from argparse import ArgumentParser
from configparser import ConfigParser
from subprocess import call

##### Define some constants ############################################################################################

DISCORD_PREFIX = '[Discord] '
COMMAND_PREFIX = '#gatekeep'
WHITELIST_COMMAND_TEMPLATE = 'tmux send-keys -t "0:0" Enter "whitelist add {}" Enter'  # Vulnerable to command injection

##### Read in our ID and secret from config ############################################################################

config = ConfigParser()
config_path = 'config.ini'
config.read(config_path)

if not config.sections():
    print('No existing config was found')
    print('Copy the following blank template into ' + config_path + ' and fill in the blanks:')
    print('[Discord]\n' +
          'client_id = \n' +
          'client_secret = \n' +
          'bot_token = \n' +
          'bot_owner = \n' +
          'bot_server = \n')
    exit(1)

if 'Discord' not in config:
    print('Failed to read config: \'Discord\' section missing')
    exit(1)
if 'client_id' not in config['Discord']:
    print('Failed to read config: \'client_id\' missing from section \'Discord\'')
    exit(1)
if 'bot_token' not in config['Discord']:
    print('Failed to read config: \'bot_token\' missing from section \'Discord\'')
    exit(1)
# if 'bot_owner' not in config['Discord']:
#     print('Failed to read config: \'bot_owner\' missing from section \'Discord\'')
#     exit(1)
if 'bot_server' not in config['Discord']:
    print('Failed to read config: \'bot_server\' missing from section \'Discord\'')
    exit(1)

discord_id = config['Discord']['client_id']
discord_bot_token = config['Discord']['bot_token']
discord_bot_server = config['Discord']['bot_server']
try:
    discord_bot_owner = config['Discord']['bot_owner']
except KeyError:
    discord_bot_owner = ''

##### Parse arguments ##################################################################################################

parser = ArgumentParser()
args = parser.parse_args()

##### Define some functions ############################################################################################

def whitelist(users: str):
    for user in users.split():
        call(WHITELIST_COMMAND_TEMPLATE.format(user))

##### Set up the Discord bot ###########################################################################################

bot = discord.Client()

@bot.event
async def on_ready():
    print(DISCORD_PREFIX + 'Bot logged in!')

@bot.event
async def on_message(message):
    # Declare this globally here, since we use it early on, /and/ in a command
    #global discord_bot_channel

    # Make sure this is from the desired server
    if str(message.guild.id) != discord_bot_server:
        # print(DISCORD_PREFIX + 'Got a message from server {} channel {}, expected server {}'.format(message.guild.id, message.channel.id, discord_bot_server))
        return

    # If the bot owner is set, make sure this is from them
    if discord_bot_owner and str(message.author.id) not in discord_bot_owner:
        # print(DISCORD_PREFIX + 'Got a message from user {}, expected user {}'.format(message.author.id, discord_bot_owner))
        return

    # Determine what prefix was used to address us, if any
    prefix = ''
    if message.content.startswith(COMMAND_PREFIX):
        prefix = COMMAND_PREFIX
    elif message.content.startswith('<@' + str(discord_id) + '>'):
        prefix = '@{}#{}'.format(bot.user.name, bot.user.discriminator)
    else:
        # We weren't addressed, we can stop here
        return

    # Define a help function
    async def help():
        message.channel.send('Command list:\n' +
                             '\n' +
                             '`help` - Shows this help text\n' +
                             '`whitelist` - Add user(s) to the whitelist')

    # Split the command into arguments
    args = message.content.strip().split()[1:]

    # Return a message if no args are provided
    if not args:
        await message.channel.send('Usage: `{} whitelist <username> [username...]`'.format(prefix), delete_after=30)
    else:
        # Filter what command came through
        if args[0] == 'help':
            await help()

        elif args[0] == 'whitelist':
            if len(args) < 1:
                await help()
            else:
                await whitelist(' '.join(args[1:]))


    # Delete the command
    await message.delete()

##### Start the Discord bot ############################################################################################

bot.run(discord_bot_token)
