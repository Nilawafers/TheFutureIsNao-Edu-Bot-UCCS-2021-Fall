import qi
import stk.runner
import stk.events
import stk.services
import stk.logging



class SpeechEvent(object):
	APP_ID = "orgs.uccs.simyan.SpeechEvent"
	def __init__(self,qiapp):
	# Component properties
		self.qiapp = qiapp
		self.events =stk.events.EventHelper(qiapp.session)
		self.s = stk.services.ServiceCache(qiapp.session)
		self.logger = stk.logging.get_logger(qiapp.session,self.APP_ID)

		self.asr =self.s.ALSpeechRecognition
		self.tts = self.s.ALTextToSpeech
		self.cid = None

	def onWordRecognized(self,value):
		self.asr.pause(True)
		self.tts.say("I heard {0}".format(value[0]))
		print(value)
		self.asr.pause(False)
		#self.logger.info("Event detected")
		#self.logger.info("\tKey: {0}".format(key))
		#self.loggger.info("\tValue: {0}".format(value))
		#self.logger.info("\tMessage: {0}".format(message))


	@qi.nobind
	def on_start(self):
		self.logger.info("Starting speech event service...")
		if hasattr (self.asr, "pause"):
			self.logger.info("Pausing ASR engine...")
			self.asr.pause(False)

		if hasattr (self.asr,"setLanuage"):
			self.logger.info("Setting language...")
			self.asr.setLanguage("English")

		if hasattr(self.asr,"setVocabulary"):
			self.logger.info("Setting vocabulary...")
			self.asr.pause(True)
			self.asr.setVocabulary(["circle","triangle","line","square"],True)
			self.asr.pause(False)
		self.logger.info("Subscribing to WordRecognized event...")
		self.cid = self.events.subscribe("WordRecognized",self.__class__.__name__,self.onWordRecognized)
		self.logger.info("Subscribed. Waiting to hear words")   

	@qi.bind(returnType = qi.Void, paramsType = [])
	def stop(self):
		self.logger.info("SpeechEvent stopped by request")
		self.qiapp.stop()

	@qi.nobind
	def on_stop(self):
		self.events.disconnect("WordRecognized",self.cid)
		self.logger.info("SpeechEvent finished")


if __name__ == "__main__":
     stk.runner.run_service(SpeechEvent)
